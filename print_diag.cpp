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
                std::cout << X[i * n + j] << " ";
            }
        }
        std::cout << "\n";
    }
}

float softdtw_stencil(float *D, float *R, uint m, uint n, float gamma)
{
    // dynamic shared memory diagonal buffer array for caching the previous
    // diagonals.
    // length is (max(m,n) + 2) * 3 because it needs to store three
    // diagonals of R and the longest diagonal is (max(m,n)+2)
    uint max_dim = max(m, n);
    printf("max_dim = %d\n", max_dim);
    float cost = 0;
    float *stencil = new float[(max_dim + 2) * 3];
    // number of antidiagonals is 2 * max(m,n) - 1
    const uint passes = 2 * max_dim - 2;

    // each pass is one diagonal of the distance matrix
    for (uint p = 0; p < passes; p++)
    {
        printf("pass %d\n", p);
        for (uint tx = 0; tx < max_dim + 2; tx++)
        {
            uint jj = max((uint)0, min(p - tx, n - 1));
            uint i = tx + 1;
            uint j = jj + 1;

            bool is_wave = tx + jj == p && tx < m + 2 && jj < n + 2;
            uint cur_idx = ((p + 2) % 3) * (max_dim + 2);
            uint prev_idx = ((p + 1) % 3) * (max_dim + 2);
            uint prev2_idx = (p % 3) * (max_dim + 2);
            if (is_wave)
            {
                if (p == 0 && tx == 0)
                {
                    stencil[prev2_idx] = 0;
                }
                printf("tid %d loading R[%d, %d] = %.2f into prev2[%d]\n", tx,
                       i - 1, j - 1, R[tx * (n + 2) + jj], jj);
                stencil[prev2_idx + jj] = R[tx * (n + 2) + jj];
            }
        }
        for (uint tx = 0; tx < max_dim + 2; tx++)
        {
            uint pp = p - 2;
            uint jj = max((uint)0, min(pp - tx, n - 1));
            uint i = tx + 1;
            uint j = jj + 1;
            bool is_wave = tx + jj == pp && i < m && j < n;
            // calculate index offsets into the shared memory array for each
            // diagonal, using mod to rotate them.
            uint cur_idx = ((pp + 2) % 3) * (max_dim + 2);
            uint prev_idx = ((pp + 1) % 3) * (max_dim + 2);
            uint prev2_idx = (pp % 3) * (max_dim + 2);
            // printf("cur_idx = %d\n", cur_idx);
            // print_matrix(&stencil[cur_idx], 1, (max(m, n) + 2));
            // printf("prev_idx = %d\n", prev_idx);
            // print_matrix(&stencil[prev_idx], 1, (max(m, n) + 2));
            // printf("prev2_idx = %d\n", prev2_idx);
            // print_matrix(&stencil[prev2_idx], 1, (max(m, n) + 2));
            if (is_wave)
            {
                // float c = D[tx * n + jj];
                float c = D[jj * n + tx];
                printf("tid %d reading %.2f from D[%d,%d]\n", tx, c, jj, tx);
                // read the elements of R from the stencil
                float r1 = stencil[prev_idx + i];
                float r2 = stencil[prev_idx + i - 1];
                float r3 = stencil[prev2_idx + i - 1];
                printf("tid %d reading %.2f from prev[%d]\n", tx, r1, i);
                printf("tid %d reading %.2f from prev[%d]\n", tx, r2, i - 1);
                printf("tid %d reading %.2f from prev2[%d]\n", tx, r3, i - 1);
                // float r1 = R[bD2 + (i - 1) * (n + 2) + j];
                // float r2 = R[bD2 + i * (n + 2) + j - 1];
                // float r3 = R[bD2 + (i - 1) * (n + 2) + j - 1];
                double prev_min = softmin(r1, r2, r3, gamma);
                // write the current element of R back to the stencil
                // R[i * (n + 2) + j] = c + prev_min;
                printf("tid %d writing cost %.2f to cur[%d]\n", tx,
                       c + prev_min, i);
                stencil[cur_idx + i] = c + prev_min;
                // printf("%.2f\n", c + prev_min);
                R[(j - 1) * (n + 2) + (i - 1)] = stencil[prev2_idx + tx];
                printf("tid %d writing %.2f to R[%d, %d]\n", tx,
                       stencil[prev2_idx + tx], j - 1, i - 1);
                if (tx == max_dim - 1)
                    cost = stencil[prev2_idx + tx];
            }
        }
        print_matrix(R, (m + 2), (n + 2));
    }
    return cost;
}

int main()
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

int _main()
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
