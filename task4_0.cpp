#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <math.h>

int* pSerialPivotPos; // The Number of pivot rows selected at the iterations
int* pSerialPivotIter; // The Iterations, at which the rows were pivots

// Function for memory allocation, definition of the objects elements and random initialization of elements
void ProcessInitialization(double*& pMatrix, double*& pVector, double*& pResult, int& Size) {
    // Memory allocation
    pMatrix = new double[Size * Size];
    pVector = new double[Size];
    pResult = new double[Size];
    // Initialization of the matrix and the vector elements
    int i, j; // Loop variables
    srand(unsigned(clock()));
    for (i = 0; i < Size; ++i) {
        pVector[i] = rand() / double(1000);
        for (j = 0; j < Size; ++j) {
            if (j <= i)
                pMatrix[i * Size + j] = rand() / double(1000);
            else
                pMatrix[i * Size + j] = 0;
        }
    }
}
// Finding the pivot row
int FindPivotRow(double* pMatrix, int Size, int Iter) {
    int PivotRow = -1; // The index of the pivot row
    int MaxValue = 0; // The value of the pivot element
    int i; // Loop variable
    // Choose the row, that stores the maximum element
    for (i = 0; i < Size; ++i) {
        if ((pSerialPivotIter[i] == -1) &&
            (fabs(pMatrix[i * Size + Iter]) > MaxValue)) {
            PivotRow = i;
            MaxValue = fabs(pMatrix[i * Size + Iter]);
        }
    }
    return PivotRow;
}
// Column elimination
void SerialColumnElimination(double* pMatrix, double* pVector, int Pivot, int Iter, int Size) {
    double PivotValue, PivotFactor;
    PivotValue = pMatrix[Pivot * Size + Iter];
    for (int i = 0; i < Size; ++i) {
        if (pSerialPivotIter[i] == -1) {
            PivotFactor = pMatrix[i * Size + Iter] / PivotValue;
            for (int j = Iter; j < Size; ++j) {
                pMatrix[i * Size + j] -= PivotFactor * pMatrix[Pivot * Size + j];
            }
            pVector[i] -= PivotFactor * pVector[Pivot];
        }
    }
}
// Gaussian elimination
void SerialGaussianElimination(double* pMatrix, double* pVector, int Size) {
    int Iter; // The number of the iteration of the Gaussian elimination
    int PivotRow; // The number of the current pivot row
    for (Iter = 0; Iter < Size; ++Iter) {
        // Finding the pivot row
        PivotRow = FindPivotRow(pMatrix, Size, Iter);
        pSerialPivotPos[Iter] = PivotRow;
        pSerialPivotIter[PivotRow] = Iter;
        SerialColumnElimination(pMatrix, pVector, PivotRow, Iter, Size);
    }
}
// Back substution
void SerialBackSubstitution(double* pMatrix, double* pVector, double* pResult, int Size) {
    int RowIndex, Row;
    for (int i = Size - 1; i >= 0; --i) {
        RowIndex = pSerialPivotPos[i];
        pResult[i] = pVector[RowIndex] / pMatrix[Size * RowIndex + i];
        for (int j = 0; j < i; ++j) {
            Row = pSerialPivotPos[j];
            pVector[Row] -= pMatrix[Row * Size + i] * pResult[i];
            pMatrix[Row * Size + i] = 0;
        }
    }
}
// Function for the execution of Gauss algorithm
void SerialResultCalculation(double* pMatrix, double* pVector, double* pResult, int Size) {
    // Memory allocation
    pSerialPivotPos = new int[Size];
    pSerialPivotIter = new int[Size];
    for (int i = 0; i < Size; ++i) {
        pSerialPivotIter[i] = -1;
    }
    // Gaussian elimination
    SerialGaussianElimination(pMatrix, pVector, Size);
    // Back substitution
    SerialBackSubstitution(pMatrix, pVector, pResult, Size);
    // Memory deallocation
    delete[] pSerialPivotPos;
    delete[] pSerialPivotIter;
}
// Function for computational process termination
void ProcessTermination(double* pMatrix, double* pVector, double* pResult) {
    delete[] pMatrix;
    delete[] pVector;
    delete[] pResult;
}

int main() {
    double* pMatrix; // The matrix of the linear system
    double* pVector; // The right parts of the linear system
    double* pResult; // The result vector
    int i, n, Sizes[] = { 10, 100, 500, 1000, 1500, 2000, 2500, 3000 }; // The sizes of the initial matrix and the vector
    clock_t start;
    printf("Serial Gauss algorithm for solving linear systems\n");
    for (i = 0; i < 8; ++i) {
        n = Sizes[i];
        // Memory allocation and definition of objects' elements
        ProcessInitialization(pMatrix, pVector, pResult, n);
        // Execution of Gauss algorithm
        start = clock();
        SerialResultCalculation(pMatrix, pVector, pResult, n);
        // Printing the execution time of Gauss method
        printf("N = %d, time of execution: %dms\n", n, clock() - start);
        // Computational process termination
        ProcessTermination(pMatrix, pVector, pResult);
    }
    return 0;
}