#include <iomanip>
#include <iostream>
#include "mpi.h"
#include <cmath>
#include <complex>
using namespace std;

#define PI 3.14159265358979323846
#define msInSec 1000

int proc_cnt, my_rank;

//Function for memory allocation and data initialization
void ProcessInitialization(complex<double>*& inputSignal, complex<double>*& outputSignal, complex<double>*& buf, int& size) {
    inputSignal = new complex<double>[size];
    outputSignal = new complex<double>[size];
    buf = new complex<double>[size];
}
//Function for computational process temination
void ProcessTermination(complex<double>*& inputSignal, complex<double>*& outputSignal, complex<double>*& buf) {
    delete[] inputSignal;
    inputSignal = NULL;
    delete[] outputSignal;
    outputSignal = NULL;
    delete[] buf;
    buf = NULL;
}
void BitReversing(complex<double>* inputSignal, complex<double>* outputSignal, complex<double>* buf, int size) {
    int bitsCount = 0, proc_cnt = 8, szcopy = size, part = size / proc_cnt, shift = 0, i;
    for (int tmp_size = size; tmp_size > 1; tmp_size >>= 1, ++bitsCount);
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
    MPI_Scatter(inputSignal, size, MPI_DOUBLE_COMPLEX, buf, size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    int lim = offsets[my_rank] + part;
    for (int ind = offsets[my_rank]; ind < lim; ++ind) {
        int mask = 1 << (bitsCount - 1);
        int revInd = 0;
        for (int i = 0; i < bitsCount; ++i) {
            bool val = ind & mask;
            revInd |= val << i;
            mask = mask >> 1;
        }
        outputSignal[ind] = buf[revInd];
    }
    MPI_Allgatherv(outputSignal, buf_szs[my_rank], MPI_DOUBLE_COMPLEX, buf, buf_szs, offsets, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    delete[] buf_szs; delete[] offsets;
}
__inline void Butterfly(complex<double>* signal, complex<double> u, int offset, int butterflySize) {
    complex<double> tem = signal[offset + butterflySize] * u;
    signal[offset + butterflySize] = signal[offset] - tem;
    signal[offset] += tem;
}
void FFTCalculation(complex<double>* signal, complex<double>* buf, int size) {
    int m = 0;
    for (int tmp_size = size; tmp_size > 1; tmp_size >>= 1, ++m);
    int* buf_szs = new int[proc_cnt], * offsets = new int[proc_cnt];
    complex<double>* src;
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
    delete[] buf_szs; delete[] offsets;
}
// FFT computation
void FFT(complex<double>* inputSignal, complex<double>* outputSignal, complex<double>* buf, int size) {
    BitReversing(inputSignal, outputSignal, buf, size);
    FFTCalculation(outputSignal, buf, size);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_cnt);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int size = 1024, width = 6;
    complex<double>* inputSignal = NULL, * outputSignal = NULL, * buf = NULL;
    ProcessInitialization(inputSignal, outputSignal, buf, size);
    double T = 1, step = PI / (size + 1) / T, step2 = 2 * step, x, x2, t, f;
    for (int i = 0; i < size; ++i) {
        f = 0, x = step2 * (i + 1.), x2 = x;
        for (int j = 1; j <= 200000; ++j, x2 += x)
            f += cos(x2) / j;
        inputSignal[i] = f;
    }
    // FFT computation
    FFT(inputSignal, outputSignal, buf, size);
    for (int i = 0; i < size; ++i) outputSignal[i] /= size >> 1;
    t = step, x = step2;
    cout << setw(4) << "ind" << " | " << setw(width) << "input" << " | " << setw(width) << "f" << " | " << "accurate\n";
    cout << fixed; cout.precision(width - 2);
    for (int i = 0; i < size; ++i, t += step, x += step2) {
        f = outputSignal[0].real() / 2, x2 = x;
        for (int k = 1; k < (size >> 1); ++k, x2 += x)
            f += outputSignal[k].real() * cos(x2) + outputSignal[k].imag() * sin(x2);
        cout << setw(4) << i + 1 << " | " << setw(width) << inputSignal[i].real() << " | "
             << setw(width) << f << " | " << setw(width) << -log(2 * sin(t)) << '\n';
    }
    // Computational process termination
    ProcessTermination(inputSignal, outputSignal, buf);
    MPI_Finalize();
    return 0;
}
