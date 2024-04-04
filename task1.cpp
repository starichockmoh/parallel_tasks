#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <ctime>
#include <cmath>

// Вариант 9
#define isParalMode false
#define f(x) pow((1 + x), - (3. / 2.))

void integral_rect(const double a, const double b, const double h, double &res) {
    int i, n = (int) ((b - a) / h);
    double sum = 0., h2 = h / 2.;
#if (isParalMode)
#pragma omp parallel for reduction(+: sum)
#endif
    for (i = 0; i < n; ++i) sum += f(a + i * h + h2);
    res = sum * h;
}

void integral_simp(const double a, const double b, const double h, double &res) {
    int i, n = (int) ((b - a) / 2. / h);
    double sum1 = 0., sum2 = 0.;
#if (isParalMode)
#pragma omp parallel for reduction(+: sum1)
#endif
    for (i = 1; i <= n; ++i) sum1 += f(a + (2. * i - 1) * h);
#if (isParalMode)
#pragma omp parallel for reduction(+: sum2)
#endif
    for (i = 1; i < n; ++i) sum2 += f(a + 2. * i * h);
    res = h * (f(a) + f(b) + 4. * sum1 + 2. * sum2) / 3.;
}

double experiment(void (*integral)(const double, const double, const double, double &), double &res) {
    double a = 1., b = 1e6, h = 0.1;
    auto stime = omp_get_wtime();
    integral(a, b, h, res); // вызов функции интегрирования
    return (omp_get_wtime() - stime);
}

void experiments(const int numbExp, void (*integral)(const double, const double, const double, double &)) {
    int i;
    double res;
    double time, min_time, max_time, avg_time;
    min_time = max_time = avg_time = experiment(integral, res); // первый запуск
    for (i = 1; i < numbExp; ++i) // оставшиеся запуски
    {
        time = experiment(integral, res);
        avg_time += time;
        if (max_time < time) max_time = time;
        else if (min_time > time) min_time = time;
    }
    printf("Время выполнения: avg: %.4fms, min: %.4fms, max: %.4fms\n", avg_time / (double) numbExp, min_time, max_time);
    printf("Ответ: %.8f\n", res);
}

int main() {
    int numbExp = 100;
    printf("Метод прямоугольников, %s режим:\n", (isParalMode) ? "многопоточный" : "однопоточный");
    experiments(numbExp, integral_rect);
    printf("Метод Симпсона, %s режим:\n", (isParalMode) ? "многопоточный" : "однопоточный");
    experiments(numbExp, integral_simp);
    return 0;
}