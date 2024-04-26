#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>
using namespace std;

#define isParalMode true
#define PI 3.14159265358979323846

//Function for computational process temination
void ProcessTermination(complex<double>*& inputSignal, complex<double>*& outputSignal) {
    delete[] inputSignal;
    inputSignal = NULL;
    delete[] outputSignal;
    outputSignal = NULL;
}
void BitReversing(complex<double>* inputSignal, complex<double>* outputSignal, int size) {
    int bitsCount = 0;
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
    const int size = 1024, width = 6;
    complex<double>* inputSignal = new complex<double>[size];
    complex<double>* outputSignal = new complex<double>[size];
    double T = 1, step = PI / (size + 1) / T, step2 = 2 * step, x, x2, t, f;
    cout << (isParalMode ? "Parallel" : "Serial") << " mode\n";
#if (isParalMode)
#pragma omp parallel for private(x, x2, f)
#endif
    for (int i = 0; i < size; ++i) {
        f = 0, x = step2 * (i + 1.), x2 = x;
        for (int j = 1; j <= 200000; ++j, x2 += x)
            f += cos(x2) / j;
        inputSignal[i] = f;
    }
    // FFT computation
    FFT(inputSignal, outputSignal, size);
    for (int i = 0; i < size; ++i) outputSignal[i] /= size >> 1;
    t = step, x = step2;
    cout << setw(4) << "ind" << " | " << setw(width) << "input" << " | " << setw(width) << "f" << " | " << "accurate\n";
    cout << fixed; cout.precision(width - 2);
    for (int i = 0; i < size; ++i, t += step, x += step2) {
        f = outputSignal[0].real() / 2, x2 = x;
        for (int k = 1; k < (size >> 1); ++k, x2 += x)
            f += outputSignal[k].real() * cos(x2) + outputSignal[k].imag() * sin(x2);
        cout << setw(4) << i + 1 << " | " << setw(width) << inputSignal[i].real() << " | "
             << setw(width) << f << " | " << setw(width) << - log(2 * sin(t)) << '\n';
    }
    // Computational process termination
    ProcessTermination(inputSignal, outputSignal);
    return 0;
}
