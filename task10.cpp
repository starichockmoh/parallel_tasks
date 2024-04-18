#include <stdio.h>
#include "mpi.h"
#include <cmath>

// Вариант 8
#define f(x) pow(2.71828182845904523536, -5. * (x))
#define msInSec 1000

int size, rank;
// MPI_Status st;

void integral_rect_seq(const double a, const double b, const double h, double& res)
{
    int i, n = (int)((b - a) / h);
    double sum = 0., h2 = h / 2.;
    for (i = 0; i < n; ++i) sum += f(a + i * h + h2);
    res = sum * h;
}

void integral_simp_seq(const double a, const double b, const double h, double& res)
{
    int i, n = (int)((b - a) / 2. / h);
    double sum1 = 0, sum2 = 0;
    for (i = 1; i <= n; ++i) sum1 += f(a + (2. * i - 1) * h);
    for (i = 1; i < n; ++i) sum2 += f(a + 2. * i * h);
    res = h * (f(a) + f(b) + 4. * sum1 + 2. * sum2) / 3.;
}

void integral_rect(const double a, const double b, const double h, double& res)
{
    int i, n = (int)((b - a) / h);
    double sum = 0., h2 = h / 2.;
    for (i = rank; i < n; i += size) sum += f(a + i * h + h2);
    MPI_Reduce(&sum, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    res *= h;
}

void integral_simp(const double a, const double b, const double h, double& res)
{
    int i, n = (int)((b - a) / 2. / h);
    double sums[] = { 0, 0 }, ress[2];
    for (i = 1 + rank; i <= n; i += size) sums[0] += f(a + (2. * i - 1) * h);
    for (i = 1 + rank; i < n; i += size) sums[1] += f(a + 2. * i * h);
    MPI_Reduce(sums, ress, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    res = h * (f(a) + f(b) + 4. * ress[0] + 2. * ress[1]) / 3.;
}

double experiment(void (*integral) (const double, const double, const double, double&), double& res)
{
    double a = 0., b = 1e6, h = 0.1, stime = MPI_Wtime();
    integral(a, b, h, res); // вызов функции интегрирования
    return (MPI_Wtime() - stime);
}

void experiments(const int numbExp, void (*integral) (const double, const double, const double, double&))
{
    int i;
    double res, time, min_time, max_time, avg_time;
    MPI_Barrier(MPI_COMM_WORLD);
    min_time = max_time = avg_time = experiment(integral, res); // первый запуск
    for (i = 1; i < numbExp; ++i) // оставшиеся запуски
    {
        MPI_Barrier(MPI_COMM_WORLD);
        time = experiment(integral, res);
        avg_time += time;
        if (max_time < time) max_time = time;
        else if (min_time > time) min_time = time;
    }
    if (!rank) {
        printf("Execution times: avg: %.0fms, min: %.0fms, max: %.0fms\n", avg_time / (double)numbExp * msInSec, min_time * msInSec, max_time * msInSec);
        printf("Integral value: %.8f\n", res);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int numbExp = 10;
    if (!rank) printf("Rectangle method sequential mode:\n");
    experiments(numbExp, integral_rect_seq);
    if (!rank) printf("Simpson's method sequential mode:\n");
    experiments(numbExp, integral_simp_seq);
    if (!rank) printf("Rectangle method parallel mode:\n");
    experiments(numbExp, integral_rect);
    if (!rank) printf("Simpson's method parallel mode:\n");
    experiments(numbExp, integral_simp);
    MPI_Finalize();
    return 0;
}
