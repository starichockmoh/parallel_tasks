#include <iomanip>
#include <iostream>
#include <cmath>
#include <complex>
#include <time.h>
#include <omp.h>
using namespace std;

#define isParalMode true
#define PI 3.14159265358979323846

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
    //Initialization of input signal elements - tests
    //DummyDataInitialization(inputSignal, size);
    //Computational experiments
    RandomDataInitialization(inputSignal, size);
}
//Function for computational process temination
void ProcessTermination(complex<double>*& inputSignal, complex<double>*& outputSignal) {
    delete[] inputSignal;
    inputSignal = NULL;
    delete[] outputSignal;
    outputSignal = NULL;
}
void BitReversing(complex<double>* inputSignal, complex<double>* outputSignal, int size) {
    int bitsCount = 0;
    //bitsCount = log2(size)
    for (int tmp_size = size; tmp_size > 1; tmp_size >>= 1, ++bitsCount);
    //ind - index in input array
    //revInd - correspondent to ind index in output array
#if (isParalMode)
#pragma omp parallel for
#endif
    for (int ind = 0; ind < size; ++ind) {
        int mask = 1 << (bitsCount - 1);
        int revInd = 0;
        for (int i = 0; i < bitsCount; ++i) { //bit-reversing
            bool val = ind & mask;
            revInd |= val << i;
            mask = mask >> 1;
        }
        outputSignal[revInd] = inputSignal[ind];
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
    for (int p = 0; p < m; ++p) {
        int butterflySize = 1 << p, butterflyOffset = butterflySize << 1;
        double coeff = PI / butterflySize;
#if (isParalMode)
#pragma omp parallel for
#endif
        for (int i = 0; i < size; i += butterflyOffset)
            for (int j = 0; j < butterflySize; ++j)
                Butterfly(signal, complex<double>(cos(-j * coeff), sin(-j * coeff)), j + i, butterflySize);
    }
}
// FFT computation
void FFT(complex<double>* inputSignal, complex<double>* outputSignal, int size) {
    BitReversing(inputSignal, outputSignal, size);
    FFTCalculation(outputSignal, size);
}

int main() {
    complex<double>* inputSignal = NULL;
    complex<double>* outputSignal = NULL;
    int Sizes[] = { 32768, 65536, 131072, 262144, 524288 };
    const int repeatCount = 16;
    double startTime;
    double duration;
    double minDuration;
    cout << "Fast Fourier Transform, " << (isParalMode ? "parallel" : "serial") << " mode\n";
    for (int size : Sizes) {
        // Memory allocation and data initialization
        ProcessInitialization(inputSignal, outputSignal, size);
        minDuration = DBL_MAX;
        for (int i = 0; i < repeatCount; ++i) {
            startTime = clock();
            // FFT computation
            FFT(inputSignal, outputSignal, size);
            duration = (clock() - startTime) / CLOCKS_PER_SEC;
            if (duration < minDuration)
                minDuration = duration;
        }
        // Computational process termination
        ProcessTermination(inputSignal, outputSignal);
        cout << setprecision(6);
        cout << "Size = " << size << ", execution time = " << minDuration << "s\n";
    }
    return 0;
}
