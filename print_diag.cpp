#include <cmath>
#include <iostream>
#include <limits>
using namespace std;

/** Take the softmin of 3 elements
 * @param a The first element
 * @param b The second element
 * @param c The third element
 * @param gamma The smoothing factor
 */
template <class T> T softmin(T a, T b, T c, T gamma)
{
    a /= -gamma;
    b /= -gamma;
    c /= -gamma;
    T max_of = max(max(a, b), c);
    T sum = exp(a - max_of) + exp(b - max_of) + exp(c - max_of);

    return -gamma * (log(sum) + max_of);
}

void print_matrix(const char *X, const uint m, const uint n)
{
    for (uint i = 0; i < m; i++)
    {
        for (uint j = 0; j < n; j++)
        {
            std::cout << X[i * n + j] << " ";
        }
        std::cout << "\n";
    }
}

void print_matrix(const float *X, const uint m, const uint n)
{
    for (uint i = 0; i < m; i++)
    {
        for (uint j = 0; j < n; j++)
        {
            std::cout << X[i * n + j] << " ";
        }
        std::cout << "\n";
    }
}

void print_diag(const char *X, const uint m, const uint n)
{
    for (uint k = 0; k <= m + n - 2; k++)
    {
        for (uint j = 0; j <= k; j++)
        {
            uint i = k - j;
            if (i < m && j < n)
            {
                uint new_j = j - (k / n * (k % n + 1));
                // std::cout << X[i * n + j] << " ";
                printf("X[%d * %d + %d] = %c ", i, n, j, X[i * n + j]);
                printf("old ij = [%d, %d] (%d), new ij = [%d, %d]\n", i, j,
                       i * n + j, k, new_j);
            }
        }
        std::cout << "\n";
    }
}

float softdtw_stencil(float *D, float *R, uint m, uint n, float gamma)
{
    uint max_dim = max(m, n);
    float cost = 0;
    float *stencil = new float[(max_dim + 2) * 3];
    const uint passes = 2 * max_dim;
    for (uint p = 0; p < passes; p++)
    {
        for (uint tx = 0; tx < max_dim + 2; tx++)
        {
            uint pp = p;
            uint jj = max((uint)0, min(pp - tx, n + 1));
            uint i = tx + 1;
            uint j = jj + 1;
            uint cur_idx = ((pp + 2) % 3) * (max_dim + 2);
            uint prev_idx = ((pp + 1) % 3) * (max_dim + 2);
            uint prev2_idx = (pp % 3) * (max_dim + 2);
            bool is_wave = tx + jj == pp && tx < m + 2 && jj < n + 2;
            if (is_wave)
            {
                if (p == 0 && tx == 0)
                {
                    stencil[prev2_idx] = 0;
                }
                stencil[prev2_idx + jj] = R[tx * (n + 2) + jj];
            }
        }
        for (uint tx = 0; tx < max_dim + 2; tx++)
        {
            uint pp = p - 2;
            uint jj = max((uint)0, min(pp - tx, n));
            uint i = tx + 1;
            uint j = jj + 1;
            uint cur_idx = ((pp + 2) % 3) * (max_dim + 2);
            uint prev_idx = ((pp + 1) % 3) * (max_dim + 2);
            uint prev2_idx = (pp % 3) * (max_dim + 2);
            bool is_wave = tx + jj == pp && (tx < m + 1 && jj < n + 1);
            if (is_wave)
            {
                float c = D[(i - 1) * n + j - 1];
                float r1 = stencil[prev_idx + i];
                float r2 = stencil[prev_idx + i - 1];
                float r3 = stencil[prev2_idx + i - 1];
                double prev_min = softmin(r1, r2, r3, gamma);
                stencil[cur_idx + i] = c + prev_min;
                printf("tid %d loading %.2f to R[%d]\n", tx,
                       stencil[prev2_idx + (i - 1)],
                       (i - 1) * (n + 2) + (j - 1));
                R[(i - 1) * (n + 2) + (j - 1)] = stencil[prev2_idx + (i - 1)];
            }
        }
    }
    cost = R[m * (n + 2) + n];
    return cost;
}

int _main()
{
    int m = 5;
    // int k = 1;
    int n = 8;
    float gamma = 0.1;
    // float *a = new float[m]{1.0, 2.0, 3.0, 3.0, 5.0};
    // float *b = new float[n]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};
    float *D = new float[m * n]{0.0,  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0, //
                                1.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, //
                                4.0,  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
                                4.0,  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
                                16.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 1.0};

    float *R = new float[(m + 2) * (n + 2)]{0.0};
    for (int i = 1; i < (m + 2) * (n + 2); i++)
    {
        R[i] = std::numeric_limits<float>::infinity();
        // R[i] = i;
    }

    float cost = softdtw_stencil(D, R, m, n, gamma);
    print_matrix(R, m + 2, n + 2);
    printf("cost = %.2f\n", cost);
    return 0;
}

int __main()
{
    const uint dim = 5;
    char ch = 'A';
    char array[dim * dim];
    for (uint i = 0; i < dim * dim; i++)
    {
        array[i] = ch++;
    }
    print_matrix((char *)&array, 5, 5);
    print_diag((char *)&array, 5, 5);
    return 0;
}

void convert_diagonal(float *D, float *DD, uint m, uint n)
{
    for (uint tx = 0; tx < m * n; tx++)
    {
        uint i = tx / n;
        uint j = tx % n;
        // printf("%d, %d\n", i, j);
        uint dest_i = i + j;
        uint dest_j = (dest_i / m) * (dest_i % m + 1);
        printf("DD[%d, %d] = D[%d, %d]\n", dest_i, dest_j, i, j);
        DD[dest_i * m + dest_j] = D[i * n + j];
    }
}

int main()
{
    const uint m = 5;
    const uint n = 8;
    float D[m * n]{0,  1, 1, 1, 1, 1, 1, 9, // dist(X[0], Y[0])
                   1,  0, 0, 0, 0, 0, 0, 4, //
                   4,  1, 1, 1, 1, 1, 1, 1, //
                   4,  1, 1, 1, 1, 1, 1, 1, //
                   16, 9, 9, 9, 9, 9, 9, 1};
    float DD[m * (m + n - 1)]{0};
    float DE[m * (m + n - 1)]{0,  0, 0, 0, 0, //
                              1,  1, 0, 0, 0, //
                              4,  0, 1, 0, 0, //
                              4,  1, 0, 1, 0, //
                              16, 1, 1, 0, 1, //
                              9,  1, 1, 0, 1, //
                              9,  1, 1, 0, 1, //
                              9,  1, 1, 0, 9, //
                              9,  1, 1, 4, 0, //
                              9,  1, 1, 0, 0, //
                              9,  1, 0, 0, 0, //
                              1,  0, 0, 0, 0};
    convert_diagonal(D, DD, m, n);
    print_matrix(DD, m + n - 1, m);
    return 0;
}
