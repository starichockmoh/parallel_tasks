#include <iomanip>
#include <iostream>
#include "mpi.h"
#include <cmath>
#include <complex>
using namespace std;

#define PI 3.14159265358979323846
#define msInSec 1000

int proc_cnt, my_rank;

//Function for simple initialization of input signal elements
void DummyDataInitialization(complex<double>* mas, int size) {
    for (int i = 0; i < size; ++i) mas[i] = 0;
    mas[size - size / 4] = 1;
}
// Function for random initialization of objects' elements
void RandomDataInitialization(complex<double>* mas, int size) {
    srand(unsigned(clock()));
    for (int i = 0; i < size; ++i)
        mas[i] = complex<double>(rand() / 1000.0, rand() / 1000.0);
}
//Function for memory allocation and data initialization
void ProcessInitialization(complex<double>*& inputSignal, complex<double>*& outputSignal, int& size) {
    inputSignal = new complex<double>[size];
    outputSignal = new complex<double>[size];
    if(!my_rank) RandomDataInitialization(inputSignal, size);
}
//Function for computational process temination
void ProcessTermination(complex<double>*& inputSignal, complex<double>*& outputSignal) {
    delete[] inputSignal;
    inputSignal = NULL;
    delete[] outputSignal;
    outputSignal = NULL;
}
void BitReversing(complex<double>* inputSignal,	complex<double>* outputSignal, int size) {
    int i = 0, j = 0;
    while (i < size) {
        if (j > i) {
            outputSignal[i] = inputSignal[j];
            outputSignal[j] = inputSignal[i];
        }
        else if (j == i) outputSignal[i] = inputSignal[i];
        int m = size >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m = m >> 1;
        }
        j += m;
        ++i;
    }
}
__inline void Butterfly(complex<double>* signal, complex<double> u, int offset, int butterflySize) {
    complex<double> tem = signal[offset + butterflySize] * u;
    signal[offset + butterflySize] = signal[offset] - tem;
    signal[offset] += tem;
}
void FFTCalculation(complex<double>* signal, int size) {
    int m = 0;
    for (int tmp_size = size; tmp_size > 1; tmp_size >>= 1, ++m);
    int* buf_szs = new int[proc_cnt], * offsets = new int[proc_cnt];
    complex<double>* buf = new complex<double>[size], * src;
    MPI_Scatter(signal, size, MPI_DOUBLE_COMPLEX, buf, size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    bool f = true;
    for (int p = 0; p < m; ++p) {
        int butterflySize = 1 << p, butterflyOffset = butterflySize << 1;
        double coeff = PI / butterflySize;
        int part = max(butterflyOffset, size / proc_cnt); part -= part % butterflyOffset;
        int offset = 0, szcopy = size;
        for (int i = 0; i < proc_cnt; ++i) {
            if (part > szcopy) { buf_szs[i] = szcopy; szcopy = 0; }
            else { buf_szs[i] = part; szcopy -= part; }
            offsets[i] = offset;
            offset += part;
        }
        if (szcopy) {
            offset = 0;
            for (int i = 0; i < proc_cnt; ++i) {
                offsets[i] += offset;
                if (szcopy) { buf_szs[i] += butterflyOffset; offset += butterflyOffset; szcopy -= butterflyOffset; }
            }
        }
        int lim = min(size, offsets[my_rank] + part);
        src = f ? buf : signal;
        for (int i = offsets[my_rank]; i < lim; i += butterflyOffset)
            for (int j = 0; j < butterflySize; ++j)
                Butterfly(src, complex<double>(cos(-j * coeff), sin(-j * coeff)), j + i, butterflySize);
        MPI_Allgatherv(src, buf_szs[my_rank], MPI_DOUBLE_COMPLEX, f ? signal : buf, buf_szs, offsets, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
        f = !f;
    }
    if (f) memcpy(signal, buf, size * sizeof(complex<double>));
    delete[] buf_szs; delete[] offsets; delete[] buf;
}
// FFT computation
void FFT(complex<double>* inputSignal, complex<double>* outputSignal, int size) {
    if (!my_rank) BitReversing(inputSignal, outputSignal, size);
    FFTCalculation(outputSignal, size);
}
void PrintSignal(complex<double>* signal, int size) {
    cout << "Result signal" << '\n';
    for (int i = 0; i < size; ++i) cout << signal[i] << '\n';
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_cnt);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    complex<double>* inputSignal = NULL;
    complex<double>* outputSignal = NULL;
    int Sizes[] = { 32768, 65536, 131072, 262144, 524288 };
    const int repeatCount = 16;
    double startTime, duration, minDuration;
    if (!my_rank) cout << "Fast Fourier Transform\n";
    for (int size : Sizes) {
        // Memory allocation and data initialization
        ProcessInitialization(inputSignal, outputSignal, size);
        minDuration = DBL_MAX;
        for (int i = 0; i < repeatCount; ++i) {
            startTime = MPI_Wtime();
            // FFT computation
            FFT(inputSignal, outputSignal, size);
            duration = (MPI_Wtime() - startTime);
            if (duration < minDuration)
                minDuration = duration;
        }
        // Result signal output
        //if (!my_rank) PrintSignal(outputSignal, size);
        // Computational process termination
        ProcessTermination(inputSignal, outputSignal);
        if (!my_rank) cout << "Size = " << size << ", execution time = " << int (minDuration * msInSec) << "ms\n";
    }
    MPI_Finalize();
    return 0;
}
