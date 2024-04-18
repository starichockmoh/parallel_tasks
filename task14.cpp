#include <stdio.h>
#include <ctime>
#include <cmath>
#include "mpi.h"

#define msInSec 1000

int proc_cnt, rank;
int* pPivotPos;
int* pPivotIter;
int* pProcInd;
int* pProcNum;

// Function for memory allocation and definition of the objects elements
void ProcessInitialization(double*& pMatrix, double*& pVector, double*& pResult, double*& pProcRows, double*& pProcVector, double*& pProcResult, int& Size, int& RowNum) {
    Size = 5;
    int i, j, RestRows; // Number of rows, that haven't been distributed yet
    RestRows = Size;
    for (i = 0; i < rank; ++i) RestRows -= RestRows / (proc_cnt - i);
    RowNum = RestRows / (proc_cnt - rank);
    pProcRows = new double[RowNum * Size];
    pProcVector = new double[RowNum];
    pProcResult = new double[RowNum];
    pPivotPos = new int[Size];
    pPivotIter = new int[RowNum];
    pProcInd = new int[proc_cnt];
    pProcNum = new int[proc_cnt];
    for (i = 0; i < RowNum; ++i) pPivotIter[i] = -1;
    if (!rank) {
        pMatrix = { new double[Size * Size]{ 1, 1, 4, 4, 9, 2, 2, 17, 17, 82, 2, 0, 3, -1, 4, 0, 1, 4, 12, 27, 1, 2, 2, 10, 0 } };
        pVector = { new double[Size] { -9, -146, -10, -26, 37 } };
        pResult = new double[Size];
    }
}
// Function for formatted matrix output
void PrintMatrix(double* pMatrix, int RowCount, int ColCount) {
    int i, j; // Loop variables
    for (i = 0; i < RowCount; ++i) {
        for (j = 0; j < ColCount; ++j)
            printf("%7.4f ", pMatrix[i * RowCount + j]);
        printf("\n");
    }
}
// Function for formatted vector output
void PrintVector(double* pVector, int Size) {
    int i;
    for (i = 0; i < Size; ++i)
        printf("%7.4f ", pVector[i]);
}
// Function for the data distribution among the processes
void DataDistribution(double* pMatrix, double* pProcRows, double* pVector, double* pProcVector, int Size, int RowNum) {
    int* pSendNum; // Number of the elements sent to the process
    int* pSendInd; // Index of the first data element sent to the process
    int RestRows = Size; // Number of rows, that have not been distributed yet
    int i;
    // Alloc memory for temporary objects
    pSendInd = new int[proc_cnt];
    pSendNum = new int[proc_cnt];
    // Define the disposition of the matrix rows for the current process
    RowNum = (Size / proc_cnt);
    pSendNum[0] = RowNum * Size;
    pSendInd[0] = 0;
    for (i = 1; i < proc_cnt; ++i) {
        RestRows -= RowNum;
        RowNum = RestRows / (proc_cnt - i);
        pSendNum[i] = RowNum * Size;
        pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
    }
    MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, pProcRows, pSendNum[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Define the disposition of the matrix rows for current process
    RestRows = Size;
    pProcInd[0] = 0;
    pProcNum[0] = Size / proc_cnt;
    for (i = 1; i < proc_cnt; ++i) {
        RestRows -= pProcNum[i - 1];
        pProcNum[i] = RestRows / (proc_cnt - i);
        pProcInd[i] = pProcInd[i - 1] + pProcNum[i - 1];
    }
    MPI_Scatterv(pVector, pProcNum, pProcInd, MPI_DOUBLE, pProcVector, pProcNum[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Free the memory
    delete[] pSendNum;
    delete[] pSendInd;
}
// Function for the Gausian elimination
void ParallelGaussianElimination(double* pProcRows, double* pProcVector, int Size, int RowNum) {
    double multiplier, MaxValue, tmp;
    int PivotPos; // Position of the pivot row in the process stripe
    struct { double MaxValue; int rank; } ProcPivot, Pivot;
    double* pPivotRow = new double[Size + 1];
    // The iterations of the Gaussian elimination stage
    for (int i = 0; i < Size; ++i) {
        // Calculating the local pivot row
        MaxValue = 0;
        for (int j = 0; j < RowNum; ++j) {
            tmp = fabs(pProcRows[j * Size + i]);
            if ((pPivotIter[j] == -1) && (MaxValue < tmp)) {
                MaxValue = tmp;
                PivotPos = j;
            }
        }
        ProcPivot.MaxValue = MaxValue;
        ProcPivot.rank = rank;
        // Find the pivot process (process with the maximum value of MaxValue)
        MPI_Allreduce(&ProcPivot, &Pivot, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        // Broadcasting the pivot row
        if (rank == Pivot.rank) {
            pPivotIter[PivotPos] = i;
            pPivotPos[i] = pProcInd[rank] + PivotPos;
        }
        MPI_Bcast(&pPivotPos[i], 1, MPI_INT, Pivot.rank, MPI_COMM_WORLD);

        if (rank == Pivot.rank) {
            for (int j = 0; j < Size; ++j)
                pPivotRow[j] = pProcRows[PivotPos * Size + j];
            pPivotRow[Size] = pProcVector[PivotPos];
        }
        MPI_Bcast(pPivotRow, Size + 1, MPI_DOUBLE, Pivot.rank, MPI_COMM_WORLD);
        for (int k = 0; k < RowNum; ++k) {
            if (pPivotIter[k] == -1) {
                multiplier = pProcRows[k * Size + i] / pPivotRow[i];
                for (int j = i; j < Size; ++j) {
                    pProcRows[k * Size + j] -= pPivotRow[j] * multiplier;
                }
                pProcVector[k] -= pPivotRow[Size] * multiplier;
            }
        }
    }
}
// Function for the back substitution
void ParallelBackSubstitution(double* pProcRows, double* pProcVector, double* pProcResult, int Size, int RowNum) {
    int IterProcRank; // Rank of the process with the current pivot row
    int IterPivotPos; // Position of the pivot row of the process
    int RowIndex;
    double IterResult; // Calculated value of the current unknown
    double val;
    // Iterations of the back substitution stage
    for (int i = Size - 1; i >= 0; --i) {
        // Calculating the rank of the process, which holds the pivot row
        RowIndex = pPivotPos[i];
        for (int i = 0; i < proc_cnt - 1; ++i)
            if ((pProcInd[i] <= RowIndex) && (RowIndex < pProcInd[i + 1]))
                IterProcRank = i;
        if (RowIndex >= pProcInd[proc_cnt - 1]) IterProcRank = proc_cnt - 1;
        IterPivotPos = RowIndex - pProcInd[IterProcRank];
        // Calculating the unknown
        if (rank == IterProcRank) {
            IterResult = pProcVector[IterPivotPos] / pProcRows[IterPivotPos * Size + i];
            pProcResult[IterPivotPos] = IterResult;
        }
        MPI_Bcast(&IterResult, 1, MPI_DOUBLE, IterProcRank, MPI_COMM_WORLD);
        for (int j = 0; j < RowNum; ++j)
            if (pPivotIter[j] < i) {
                val = pProcRows[j * Size + i] * IterResult;
                pProcVector[j] = pProcVector[j] - val;
            }
    }
}
// Function for the execution of the parallel Gauss algorithm
void ParallelResultCalculation(double* pMatrix, double* pVector, double* pProcRows, double* pProcVector, double* pProcResult, double* pResult, int Size, int RowNum) {
    DataDistribution(pMatrix, pProcRows, pVector, pProcVector, Size, RowNum);
    ParallelGaussianElimination(pProcRows, pProcVector, Size, RowNum);
    ParallelBackSubstitution(pProcRows, pProcVector, pProcResult, Size, RowNum);
    MPI_Gatherv(pProcResult, pProcNum[rank], MPI_DOUBLE, pResult, pProcNum, pProcInd, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
// Function for computational process termination
void ProcessTermination(double* pMatrix, double* pVector, double* pResult, double* pProcRows, double* pProcVector, double* pProcResult) {
    if (!rank) {
        delete[] pMatrix;
        delete[] pVector;
        delete[] pResult;
    }
    delete[] pProcRows;
    delete[] pProcVector;
    delete[] pProcResult;
    delete[] pPivotPos;
    delete[] pPivotIter;
    delete[] pProcInd;
    delete[] pProcNum;
}

void main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_cnt);
    double* pMatrix; // Matrix of the linear system
    double* pVector; // Right parts of the linear system
    double* pResult; // Result vector
    double* pProcRows; // Rows of the matrix A
    double* pProcVector; // Block of the vector b
    double* pProcResult; // Block of the vector x
    int RowNum; // Number of the matrix rows
    double start;
    int i, Size;
    if (!rank) printf("Gauss algorithm for solving linear systems\n");
    // Memory allocation and definition of objects' elements
    ProcessInitialization(pMatrix, pVector, pResult, pProcRows, pProcVector, pProcResult, Size, RowNum);
    // The matrix and the vector output
    if (!rank) {
        printf("Initial Matrix\n");
        PrintMatrix(pMatrix, Size, Size);
        printf("Initial Vector\n");
        PrintVector(pVector, Size);
    }
    // Execution of Gauss algorithm
    start = MPI_Wtime();
    ParallelResultCalculation(pMatrix, pVector, pProcRows, pProcVector, pProcResult, pResult, Size, RowNum);
    // Printing the execution time of Gauss method
    if (!rank) {
        printf("\nTime of execution: %dms\n", Size, int((MPI_Wtime() - start) * msInSec));
        // Printing the result vector
        printf("Result Vector:\n");
        PrintVector(pResult, Size);
    }
    // Computational process termination
    ProcessTermination(pMatrix, pVector, pResult, pProcRows, pProcVector, pProcResult);
    MPI_Finalize();
}
