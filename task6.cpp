#include <stdio.h>
#include <omp.h>
#include <ctime>
#include <cmath>

#define isParalMode true

// Function for simple initialization of the matrix and the vector elements
void DummyDataInitialization(double* pMatrix, double* pVector, int Size) {
    int i, j; // Loop variables
    for (i = 0; i < Size; ++i) {
        pVector[i] = i + 1;
        for (j = 0; j < Size; ++j) {
            if (j <= i)
                pMatrix[i * Size + j] = 1;
            else
                pMatrix[i * Size + j] = 0;
        }
    }
}
// Function for random initialization of the matrix and the vector elements
void RandomDataInitialization(double* pMatrix, double* pVector, int Size) {
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
// Function for memory allocation and definition of the objects elements
void ProcessInitialization(double*& pMatrix, double*& pVector, double*& pResult, int& Size, const bool isRandomInit = false) {
    // Memory allocation
    pMatrix = new double[Size * Size];
    pVector = new double[Size];
    pResult = new double[Size];
    // Initialization of the matrix and the vector elements
    if (isRandomInit) RandomDataInitialization(pMatrix, pVector, Size);
    else DummyDataInitialization(pMatrix, pVector, Size);
}
// Upper relaxation method
void UpperRelaxationMethod(double* pMatrix, double* pVector, double* pResult, int& Size,
                           const double eps = 1e-6, const double omega = 1.5, const int maxIter = 100) {
    int i, j, iter = 0;
    double prevResult, accum, norm;
#if (isParalMode)
#pragma omp parallel for
#endif
    for (i = 0; i < Size; ++i)
        pResult[i] = pVector[i]; // initial approximation
    do {
        if (iter == maxIter) return;
        ++iter;
        norm = 0;
        for (i = 0; i < Size; ++i) {
            accum = 0;
            prevResult = pResult[i];
#if (isParalMode)
#pragma omp parallel for reduction(+: accum)
#endif
            for (j = 0; j < Size; ++j)
                accum += pMatrix[i * Size + j] * pResult[j];
            pResult[i] += omega * (pVector[i] - accum) / pMatrix[i * Size + i];
            accum = abs(pResult[i] - prevResult); // "accum" uses as tmp
            norm = norm > accum ? norm : accum;
        }
    } while (norm > eps);
}
// Function for formatted matrix output
void PrintMatrix(double* pMatrix, int RowCount, int ColCount) {
    int i, j; // Loop variables
    for (i = 0; i < RowCount; i++) {
        for (j = 0; j < ColCount; j++)
            printf("%7.4f ", pMatrix[i * RowCount + j]);
        printf("\n");
    }
}
// Function for formatted vector output
void PrintVector(double* pVector, int Size) {
    int i;
    for (i = 0; i < Size; i++)
        printf("%7.4f ", pVector[i]);
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
    printf("%s upper relaxation method algorithm for solving linear systems\n", isParalMode ? "Parallel" : "Serial");
    for (i = 0; i < 8; ++i) {
        n = Sizes[i];
        // Memory allocation and definition of objects' elements
        ProcessInitialization(pMatrix, pVector, pResult, n);
        // Execution of Gauss algorithm
        start = clock();
        UpperRelaxationMethod(pMatrix, pVector, pResult, n);
        // Printing the execution time of Gauss method
        printf("N = %d, time of execution: %dms\n", n, clock() - start);
        // Computational process termination
        ProcessTermination(pMatrix, pVector, pResult);
    }
    //int Size = 5;
    //ProcessInitialization(pMatrix, pVector, pResult, Size);
    //printf("Initial Matrix\n");
    //PrintMatrix(pMatrix, Size, Size);
    //printf("Initial Vector\n");
    //PrintVector(pVector, Size);
    //UpperRelaxationMethod(pMatrix, pVector, pResult, Size);
    //printf("\nResult Vector:\n");
    //PrintVector(pResult, Size);
    // Computational process termination
    // ProcessTermination(pMatrix, pVector, pResult);
    return 0;
}
