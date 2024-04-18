#include <stdio.h>
#include "mpi.h"
#include <cmath>

// Вариант 2 - метод трапеций
#define PI 3.1415926535897932384626433832795
#define f(x, y) (pow(2.71828182845904523536, sin(PI * (x)) * cos(PI * (y))) + 1.)
#define msInSec 1000

int size, rank;

void integral_paral(const double a1, const double b1, const double a2, const double b2, const int n, double& res)
{
    int i, j;
    double sum = 0., ba1 = (b1 - a1), ba2 = (b2 - a2), h1 = ba1 / n, h2 = ba2 / n;
    // воспользуемся формулой Котеса
    for (i = 1 + rank; i < n; i += size)
        for (j = 1; j < n; ++j)
            sum += f(a1 + i * h1, a2 + j * h2);
    MPI_Reduce(&sum, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    res = h1 * h2 * ((f(a1, a2) + f(a1 + n * h1, a2 + n * h2)) / 2. + res) / ba1 / ba2;
}

double experiment(void (*integral) (const double, const double, const double, const double, const int, double&), double& res)
{
    double a1 = 0., b1 = 16., a2 = 0., b2 = 16., stime = MPI_Wtime();
    int n = 5000;
    integral(a1, b1, a2, b2, n, res); // вызов функции интегрирования
    return (MPI_Wtime() - stime);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int i, numbExp = 10;
    double res, time, min_time, max_time, avg_time;
    MPI_Barrier(MPI_COMM_WORLD);
    min_time = max_time = avg_time = experiment(integral_paral, res); // первый запуск
    for (i = 1; i < numbExp; ++i) // оставшиеся запуски
    {
        MPI_Barrier(MPI_COMM_WORLD);
        time = experiment(integral_paral, res);
        avg_time += time;
        if (max_time < time) max_time = time;
        else if (min_time > time) min_time = time;
    }
    if (!rank) {
        printf("Execution times: avg: %.0fms, min: %.0fms, max: %.0fms\n", avg_time / (double)numbExp * msInSec, min_time * msInSec, max_time * msInSec);
        printf("Integral value: %.8f\n", res);
    }
    MPI_Finalize();
    return 0;
}
