#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>


// Kernel to compute force matrix
__global__ void computeCollisions(double* xPos, double* yPos, int N, double* radii) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure i and j are valid particle indices and avoid self-interaction
    if (i < N && j < N && i != j) {
        // Load particle positions
        float x1 = xPos[i];
        float y1 = yPos[i];
        float x2 = xPos[j];
        float y2 = yPos[j];

        // Compute vector between particles and distance squared
        float dx = x2 - x1;
        float dy = y2 - y1;
        float dist2 = dx * dx + dy * dy;

        // Define constants
        const float response_coef = 0.5;
        const float eps = 0.0000000001;  

        // Check for overlap and process if within interaction range
        if (dist2 < 0.7f && dist2 > eps) { // Assuming radius is 1.0 for simplicity
            float sep = sqrtf(dist2);

            // Calculate overlap correction
            float overlap = -response_coef * 0.5 * sep; // Total radius is 2.0
            float colVecX = (dx / sep) * overlap;
            float colVecY = (dy / sep) * overlap;

            // Apply corrections
            xPos[i] += colVecX;
            yPos[i] += colVecY;
            xPos[j] += -colVecX;
            yPos[j] += -colVecY;

        }
    }
}


// Verlet integration kernel
__global__ void integratePositions(int count, double* dev_xPosMatrix, double* dev_yPosMatrix, double* xPos, double* yPos,
    double* xVel, double* yVel, int N, double timeStep, double* radii, double bw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int sample = 100;
    double margin = 0.5;
    double x_min = margin;
    double x_max = bw - margin;
    double y_min = margin;
    double y_max = bw - margin;

    if (i < N) {
        if (count % sample == 0) {
            int row = count / sample;
            dev_xPosMatrix[row * N + i] = xPos[i];
            dev_yPosMatrix[row * N + i] = yPos[i];
        }

        xPos[i] += xVel[i] * timeStep;
        yPos[i] += yVel[i] * timeStep + 0.5 * -0.05 * timeStep * timeStep;

        if (xPos[i] > x_max) {
            xPos[i] = x_max;
            //xVel[i] = -xVel[i];
        }
        else if (xPos[i] < x_min) {
            xPos[i] = x_min;
            //xVel[i] = -xVel[i];
        }

        if (yPos[i] > y_max) {
            yPos[i] = y_max;
            //yVel[i] = -yVel[i];
        }
        else if (yPos[i] < y_min) {
            yPos[i] = y_min;
        }
    }
}

// Verlet velocity integration kernel
__global__ void integrateVelocities(double* xVel, double* yVel, int N, double timeStep) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        yVel[i] += -0.05 * timeStep;
    }
}

// Function to write a matrix to a CSV file
void writeMatrixToFile(double* matrix, int rows, int cols, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s for writing.\n", filename);
        return;
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%f", matrix[i * cols + j]);
            if (j < cols - 1) fprintf(file, ","); // No trailing comma at the end of the row
        }
        fprintf(file, "\n"); // New line at the end of each row
    }
    fclose(file);
}

void printMatrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    const double timeStep = 1e-4;
    const int runTime = 10;
    int samplerate = 100;
    const int N = 2000;
    const double boxwidth = 44.0;

    const int iterations = runTime / timeStep;

    // Allocate memory
    double* xPos = (double*)malloc(N * sizeof(double));
    double* yPos = (double*)malloc(N * sizeof(double));
    double* xVel = (double*)malloc(N * sizeof(double));
    double* yVel = (double*)malloc(N * sizeof(double));
    double* radii = (double*)malloc(N * sizeof(double));
    double* xPositionMatrix = (double*)malloc((iterations / samplerate) * N * sizeof(double));
    double* yPositionMatrix = (double*)malloc((iterations / samplerate) * N * sizeof(double));

    // Initialize positions, velocities, etc.
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        xPos[i] = (double)rand() / (double)(RAND_MAX / (boxwidth - 1)); 
        yPos[i] = (double)rand() / (double)(RAND_MAX / (boxwidth - 1));
        xVel[i] = ((double)rand() / (double)(RAND_MAX / 2)) - (2 / 2);
        yVel[i] = ((double)rand() / (double)(RAND_MAX / 2)) - (2 / 2);
        radii[i] = 0.15;
    }

    // Allocate device memory
    double* dev_xPos, * dev_yPos, * dev_xVel, * dev_yVel, * dev_radii;
    double* dev_xmat, * dev_ymat;

    cudaMalloc((void**)&dev_xPos, N * sizeof(double));
    cudaMalloc((void**)&dev_yPos, N * sizeof(double));
    cudaMalloc((void**)&dev_xVel, N * sizeof(double));
    cudaMalloc((void**)&dev_yVel, N * sizeof(double));
    cudaMalloc((void**)&dev_radii, N * sizeof(double));
    cudaMalloc((void**)&dev_xmat, (iterations / samplerate) * N * sizeof(double));
    cudaMalloc((void**)&dev_ymat, (iterations / samplerate) * N * sizeof(double));

    // Copy data to device
    cudaMemcpy(dev_xPos, xPos, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_yPos, yPos, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_xVel, xVel, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_yVel, yVel, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_radii, radii, N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Main loop
    for (int count = 0; count < iterations; count++) {

        integratePositions << <blocksPerGrid, threadsPerBlock >> > (count, dev_xmat, dev_ymat, dev_xPos, dev_yPos,
            dev_xVel, dev_yVel, N, timeStep, dev_radii, boxwidth);

        cudaDeviceSynchronize();
        for (int i = 0; i < 3; i++) {
            computeCollisions <<<blocksPerGrid, threadsPerBlock >>> (dev_xPos, dev_yPos, N, dev_radii);
            cudaDeviceSynchronize();
        }
        integrateVelocities <<<blocksPerGrid.x, threadsPerBlock.x>>> (dev_xVel, dev_yVel, N, timeStep);
        cudaDeviceSynchronize();
    }

  

    // Copy results back to host
    cudaMemcpy(xPositionMatrix, dev_xmat, (iterations / samplerate) * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(yPositionMatrix, dev_ymat, (iterations / samplerate) * N * sizeof(double), cudaMemcpyDeviceToHost);

    writeMatrixToFile(xPositionMatrix, iterations / samplerate, N, "xPositionMatrix.csv");
    writeMatrixToFile(yPositionMatrix, iterations / samplerate, N, "yPositionMatrix.csv");


    // Free memory
    free(xPos);
    free(yPos);
    free(xVel);
    free(yVel);
    free(radii);
    free(xPositionMatrix);
    free(yPositionMatrix);

    cudaFree(dev_xPos);
    cudaFree(dev_yPos);
    cudaFree(dev_xVel);
    cudaFree(dev_yVel);
    cudaFree(dev_radii);
    cudaFree(dev_xmat);
    cudaFree(dev_ymat);
    return 0;
}
