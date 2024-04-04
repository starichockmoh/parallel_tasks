#include <stdio.h>
#include <omp.h>
#include <ctime>
#include <cmath>

//Вариант 1
int* pPivotPos; // The number of pivot rows selected at the iterations
int* pPivotIter; // The iterations, at which the rows were pivots

typedef struct {
    int PivotRow;
    double MaxValue;
} TThreadPivotRow;

// Function for memory allocation and definition of the objects elements
void ProcessInitialization(double*& pMatrix, double*& pVector, double*& pResult, int& Size) {
    // Memory allocation
    Size = 5;
    pMatrix = { new double[Size * Size]{ 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 1, 3, 6, 10, 15, 1, 4, 10, 20, 35, 1, 5, 15, 35, 70 } };
    pVector = { new double[Size]{ 15, 35, 70, 126, 210 } };
    pResult = new double[Size];
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
// Finding the pivot row
int ParallelFindPivotRow(double* pMatrix, int Size, int Iter) {
    int PivotRow = -1; // The index of the pivot row
    double MaxValue = 0; // The value of the pivot element
    int i; // Loop variable
    // Choose the row, that stores the maximum element
#pragma omp parallel
    {
        TThreadPivotRow ThreadPivotRow;
        ThreadPivotRow.MaxValue = 0;
        ThreadPivotRow.PivotRow = -1;
#pragma omp for
        for (i = 0; i < Size; ++i) {
            if ((pPivotIter[i] == -1) &&
                (fabs(pMatrix[i * Size + Iter]) > ThreadPivotRow.MaxValue)) {
                ThreadPivotRow.PivotRow = i;
                ThreadPivotRow.MaxValue = fabs(pMatrix[i * Size + Iter]);
            }
        }
#pragma omp critical
        {
            if (ThreadPivotRow.MaxValue > MaxValue) {
                MaxValue = ThreadPivotRow.MaxValue;
                PivotRow = ThreadPivotRow.PivotRow;
            }
        } // pragma omp critical
    } // pragma omp parallel
    return PivotRow;
}
// Column elimination
void ParallelColumnElimination(double* pMatrix, double* pVector, int Pivot, int Iter, int Size) {
    double PivotValue, PivotFactor;
    PivotValue = pMatrix[Pivot * Size + Iter];
#pragma omp parallel for private(PivotFactor) schedule(dynamic, 1)
    for (int i = 0; i < Size; ++i) {
        if (pPivotIter[i] == -1) {
            PivotFactor = pMatrix[i * Size + Iter] / PivotValue;
            for (int j = Iter; j < Size; ++j) {
                pMatrix[i * Size + j] -= PivotFactor * pMatrix[Pivot * Size + j];
            }
            pVector[i] -= PivotFactor * pVector[Pivot];
        }
    }
}
// Gaussian elimination
void ParallelGaussianElimination(double* pMatrix, double* pVector, int Size) {
    int Iter; // The number of the iteration of the Gaussian elimination
    int PivotRow; // The number of the current pivot row
    for (Iter = 0; Iter < Size; ++Iter) {
        // Finding the pivot row
        PivotRow = ParallelFindPivotRow(pMatrix, Size, Iter);
        pPivotPos[Iter] = PivotRow;
        pPivotIter[PivotRow] = Iter;
        ParallelColumnElimination(pMatrix, pVector, PivotRow, Iter, Size);
    }
}
// Back substation
void ParallelBackSubstitution(double* pMatrix, double* pVector, double* pResult, int Size) {
    int RowIndex, Row;
    for (int i = Size - 1; i >= 0; --i) {
        RowIndex = pPivotPos[i];
        pResult[i] = pVector[RowIndex] / pMatrix[Size * RowIndex + i];
#pragma omp parallel for private (Row)
        for (int j = 0; j < i; ++j) {
            Row = pPivotPos[j];
            pVector[Row] -= pMatrix[Row * Size + i] * pResult[i];
            pMatrix[Row * Size + i] = 0;
        }
    }
}
void ParallelResultCalculation(double* pMatrix, double* pVector, double* pResult, int Size) {
    // Memory allocation
    pPivotPos = new int[Size];
    pPivotIter = new int[Size];
    for (int i = 0; i < Size; ++i)
        pPivotIter[i] = -1;
    // Gaussian elimination
    ParallelGaussianElimination(pMatrix, pVector, Size);
    // Back substitution
    ParallelBackSubstitution(pMatrix, pVector, pResult, Size);
    // Memory deallocation
    delete[] pPivotPos;
    delete[] pPivotIter;
}
// Function for testing the result
void TestResult(double* pMatrix, double* pVector,
                double* pResult, int Size) {
    // Buffer for storing the vector, that is a result of multiplication of the linear system matrix by the vector of unknowns
    double* pRightPartVector;
    // Flag, that shows wheather the right parts vectors are identical or not
    int equal = 0;
    double Accuracy = 1.e-6; // Comparison accuracy
    pRightPartVector = new double[Size];
    for (int i = 0; i < Size; ++i) {
        pRightPartVector[i] = 0;
        for (int j = 0; j < Size; ++j) {
            pRightPartVector[i] += pMatrix[i * Size + j] * pResult[j];
        }
    }
    for (int i = 0; i < Size; ++i)
        if (fabs(pRightPartVector[i] - pVector[i]) > Accuracy)
            equal = 1;
    if (equal == 1)
        printf("\nThe result of the parallel Gauss algorithm is NOT correct. Check your code.");
    else
        printf("\nThe result of the parallel Gauss algorithm is correct.");
    delete[] pRightPartVector;
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
    int i, Size; // The sizes of the initial matrix and the vector
    clock_t start;
    printf("Parallel Gauss algorithm for solving linear systems\n");
    // Memory allocation and definition of objects' elements
    ProcessInitialization(pMatrix, pVector, pResult, Size);
    // The matrix and the vector output
    printf("Initial Matrix\n");
    PrintMatrix(pMatrix, Size, Size);
    printf("Initial Vector\n");
    PrintVector(pVector, Size);
    // Execution of Gauss algorithm
    start = clock();
    ParallelResultCalculation(pMatrix, pVector, pResult, Size);
    // Printing the execution time of Gauss method
    printf("\nTime of execution: %dms\n", clock() - start);
    // Printing the result vector
    printf("Result Vector:\n");
    PrintVector(pResult, Size);
    TestResult(pMatrix, pVector, pResult, Size);
    // Computational process termination
    ProcessTermination(pMatrix, pVector, pResult);
    return 0;
}