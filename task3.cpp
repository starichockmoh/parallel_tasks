#include <stdio.h>
#include <omp.h>
#include <ctime>
#include <cmath>

// Вариант 2 - метод трапеций
#define PI 3.1415926535897932384626433832795
#define f(x, y) (pow(2.71828182845904523536, sin(PI * (x)) * cos(PI * (y))) + 1.)

void integral_paral(const double a1, const double b1, const double a2, const double b2, const int n, double& res)
{
    int i, j;
    double sum = 0., ba1 = (b1 - a1), ba2 = (b2 - a2), h1 = ba1 / n, h2 = ba2 / n;
    // воспользуемся формулой Котеса
#pragma omp parallel for private(j) reduction(+: sum)
    for (i = 1; i < n; ++i)
        for (j = 1; j < n; ++j)
            sum += f(a1 + i * h1, a2 + j * h2);
    res = h1 * h2 * ((f(a1, a2) + f(a1 + n * h1, a2 + n * h2)) / 2. + sum) / ba1 / ba2;
}

clock_t experiment(void (*integral) (const double, const double, const double, const double, const int, double&), double& res)
{
    double a1 = 0., b1 = 16., a2 = 0., b2 = 16.;
    int n = 5000;
    clock_t stime = clock();
    integral(a1, b1, a2, b2, n, res); // вызов функции интегрирования
    return (clock() - stime);
}

int main()
{
    int i, numbExp = 10;
    double res;
    clock_t time, min_time, max_time, avg_time;
    min_time = max_time = avg_time = experiment(integral_paral, res); // первый запуск
    for (i = 1; i < numbExp; ++i) // оставшиеся запуски
    {
        time = experiment(integral_paral, res);
        avg_time += time;
        if (max_time < time) max_time = time;
        else if (min_time > time) min_time = time;
    }
    printf("Время выполнения: avg: %.1fms, min: %dms, max: %dms\n", avg_time / (double)numbExp, min_time, max_time);
    printf("Ответ: %.8f\n", res);
    return 0;
}