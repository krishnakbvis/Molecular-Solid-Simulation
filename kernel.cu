#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>


// Kernel to compute force matrix
__global__ void computeCollisions(float* xPos, float* yPos, int N, float* radii) {
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
        const float response_coef = 0.5f;
        const float eps = 0.0000000001f;  

        float sep = sqrtf(dist2);

        // Calculate overlap correction
        float overlap = -response_coef * 0.5 * sep; // Total radius is 2.0
        float colVecX = (dx / sep) * overlap;
        float colVecY = (dy / sep) * overlap;

        // Check for overlap and process if within interaction range
        if (dist2 < 0.7f && dist2 > eps) { // Assuming radius is 1.0 for simplicity
            // Apply corrections
            xPos[i] += colVecX;
            yPos[i] += colVecY;
            xPos[j] += -colVecX;
            yPos[j] += -colVecY;
        }
    }
}


// Verlet integration kernel
__global__ void integratePositions(int count, float* dev_xPosMatrix, float* dev_yPosMatrix, float* xPos, float* yPos,
    float* xVel, float* yVel, int N, float timeStep, float* radii, float bw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int sample = 100;
    float margin = 0.5;
    float x_min = margin;
    float x_max = bw - margin;
    float y_min = margin;
    float y_max = bw - margin;

    if (i < N) {
        if (count % sample == 0) {
            int row = count / sample;
            dev_xPosMatrix[row * N + i] = xPos[i];
            dev_yPosMatrix[row * N + i] = yPos[i];
        }

        xPos[i] += xVel[i] * timeStep;
        yPos[i] += yVel[i] * timeStep - 0.5 * 3 * timeStep * timeStep;

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
__global__ void integrateVelocities(float* xVel, float* yVel, int N, float timeStep) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        yVel[i] += -3*timeStep;
    }
}

// Function to write a matrix to a CSV file
void writeMatrixToFile(float* matrix, int rows, int cols, const char* filename) {
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

void printMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    const float timeStep = 1e-4;
    const int runTime = 10;
    int samplerate = 100;
    const int N = 3000;
    const float boxwidth = 55.0;

    const int iterations = runTime / timeStep;

    // Allocate memory
    float* xPos = (float*)malloc(N * sizeof(float));
    float* yPos = (float*)malloc(N * sizeof(float));
    float* xVel = (float*)malloc(N * sizeof(float));
    float* yVel = (float*)malloc(N * sizeof(float));
    float* radii = (float*)malloc(N * sizeof(float));
    float* xPositionMatrix = (float*)malloc((iterations / samplerate) * N * sizeof(float));
    float* yPositionMatrix = (float*)malloc((iterations / samplerate) * N * sizeof(float));

    // Initialize positions, velocities, etc.
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        xPos[i] = (float)rand() / (float)(RAND_MAX / (boxwidth - 1)); 
        yPos[i] = (float)rand() / (float)(RAND_MAX / (boxwidth - 1));
        xVel[i] = ((float)rand() / (float)(RAND_MAX / 5)) - (5 / 2);
        yVel[i] = ((float)rand() / (float)(RAND_MAX / 5)) - (5 / 2);
        radii[i] = 0.15;
    }

    // Allocate device memory
    float* dev_xPos, * dev_yPos, * dev_xVel, * dev_yVel, * dev_radii;
    float* dev_xmat, * dev_ymat;

    cudaMalloc((void**)&dev_xPos, N * sizeof(float));
    cudaMalloc((void**)&dev_yPos, N * sizeof(float));
    cudaMalloc((void**)&dev_xVel, N * sizeof(float));
    cudaMalloc((void**)&dev_yVel, N * sizeof(float));
    cudaMalloc((void**)&dev_radii, N * sizeof(float));
    cudaMalloc((void**)&dev_xmat, (iterations / samplerate) * N * sizeof(float));
    cudaMalloc((void**)&dev_ymat, (iterations / samplerate) * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(dev_xPos, xPos, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_yPos, yPos, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_xVel, xVel, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_yVel, yVel, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_radii, radii, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16); // Ceil(N / 16) in both dimensions

    int threadsPerBlock2 = 512;
    int numBlocks2 = (N + threadsPerBlock2 - 1) / threadsPerBlock2; // Round up



    // Main loop
    for (int count = 0; count < iterations; count++) {

        integratePositions << <numBlocks2, threadsPerBlock2>> > (count, dev_xmat, dev_ymat, dev_xPos, dev_yPos,
            dev_xVel, dev_yVel, N, timeStep, dev_radii, boxwidth);

        cudaDeviceSynchronize();
        computeCollisions << <numBlocks, threadsPerBlock>> > (dev_xPos, dev_yPos, N, dev_radii);
        integrateVelocities <<<numBlocks2, threadsPerBlock2>>> (dev_xVel, dev_yVel, N, timeStep);
        cudaDeviceSynchronize();
    }

  

    // Copy results back to host
    cudaMemcpy(xPositionMatrix, dev_xmat, (iterations / samplerate) * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(yPositionMatrix, dev_ymat, (iterations / samplerate) * N * sizeof(float), cudaMemcpyDeviceToHost);

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
