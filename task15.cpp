#include <stdio.h>
#include <omp.h>
#include <ctime>
#include <cmath>
#include "mpi.h"
#include <string.h>

#define msInSec 1000

int proc_cnt, rank;

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
    if (!rank) {
        if (isRandomInit) RandomDataInitialization(pMatrix, pVector, Size);
        else DummyDataInitialization(pMatrix, pVector, Size);
    }
}
// Upper relaxation method
void UpperRelaxationMethod(double* pMatrix, double* pVector, double* pResult, int& Size,
                           const double eps = 1e-6, const double omega = 1.5, const int maxIter = 100) {
    int i, j, iter = 0, szcopy = Size, part = Size / proc_cnt, shift = 0;
    double prevResult, sum, accum, norm;
    int* buf_szs = new int[proc_cnt], * offsets = new int[proc_cnt];
    for (i = 0; i < proc_cnt; ++i) {
        buf_szs[i] = part;
        offsets[i] = shift;
        shift += part;
        szcopy -= part;
    }
    if (szcopy) {
        i = 0;
        while (i < szcopy) { buf_szs[i] += 1; offsets[i] += i; ++i; }
        while (i < proc_cnt) { offsets[i] += szcopy; ++i; }
    }
    int Sz2 = Size * Size, lim = offsets[rank] + part;
    double* buf = new double[Sz2];
    if (!rank) memcpy(buf, pMatrix, Sz2 * sizeof(double));
    MPI_Scatter(buf, Sz2, MPI_DOUBLE, pMatrix, Sz2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (!rank) memcpy(buf, pVector, Size * sizeof(double));
    MPI_Scatter(buf, Size, MPI_DOUBLE, pVector, Size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (i = offsets[rank]; i < lim; ++i) buf[i] = pVector[i]; // initial approximation
    MPI_Allgatherv(buf, buf_szs[rank], MPI_DOUBLE, pResult, buf_szs, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
    do {
        if (iter == maxIter) return;
        ++iter;
        norm = 0;
        for (i = 0; i < Size; ++i) {
            accum = 0, sum = 0;
            prevResult = pResult[i];
            for (j = rank; j < Size; j += proc_cnt) accum += pMatrix[i * Size + j] * pResult[j];
            MPI_Allreduce(&sum, &accum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            pResult[i] += omega * (pVector[i] - accum) / pMatrix[i * Size + i];
            accum = abs(pResult[i] - prevResult); // "accum" uses as tmp
            norm = norm > accum ? norm : accum;
        }
    } while (norm > eps);
    delete[] buf;
}
// Function for computational process termination
void ProcessTermination(double* pMatrix, double* pVector, double* pResult) {
    delete[] pMatrix;
    delete[] pVector;
    delete[] pResult;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_cnt);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double* pMatrix; // The matrix of the linear system
    double* pVector; // The right parts of the linear system
    double* pResult; // The result vector
    int i, n, Sizes[] = { 10, 100, 500, 1000, 1500, 2000, 2500, 3000 }; // The sizes of the initial matrix and the vector
    double start;
    if (!rank) printf("Upper relaxation method algorithm for solving linear systems\n");
    for (i = 0; i < 8; ++i) {
        n = Sizes[i];
        // Memory allocation and definition of objects' elements
        ProcessInitialization(pMatrix, pVector, pResult, n);
        // Execution of Gauss algorithm
        start = MPI_Wtime();
        UpperRelaxationMethod(pMatrix, pVector, pResult, n);
        // Printing the execution time of Gauss method
        if (!rank) printf("N = %d, time of execution: %dms\n", n, int((MPI_Wtime() - start) * msInSec));
        // Computational process termination
        ProcessTermination(pMatrix, pVector, pResult);
    }
    MPI_Finalize();
    return 0;
}
