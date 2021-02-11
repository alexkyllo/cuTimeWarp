/**
 * @file timewarp.cpp
 */
#include <iostream>
#include <limits>
/**
 * @param a The first series array
 * @param b The second series array
 * @param m Length of array a
 * @param n Length of array b
 */
template <class T> T timewarp(T *a, T *b, int m, int n)
{
    // Create an m*n matrix for the warp path
    // and initialize it to infinite distances
    T *w = new T[m * n];
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            w[i * n + j] = std::numeric_limits<T>::infinity();
        }
    }
    w[0] = 0;

    // Iterate over each cell of the matrix to compute
    // lowest cost path through the preceding neighbor cells
    for (int i = 1; i < m; i++)
    {
        for (int j = 1; j < n; j++)
        {
            T cost = std::abs(a[i] - b[j]);
            double prev_min =
                std::min(w[(i - 1) * n + j],
                         std::min(w[i * n + j - 1], w[(i - 1) * n + j - 1]));
            w[i * n + j] = cost + prev_min;
        }
    }
    // Return the total cost of the warp path
    T path_cost = w[m * n - 1];
    delete[] w;
    return path_cost;
}

int main()
{
    int a[5] = {1, 2, 3, 3, 5};
    int b[8] = {1, 2, 2, 2, 2, 2, 2, 4};
    int d = timewarp<int>(a, b, 5, 8);
    // Should print: 3
    std::cout << "DTW distance: " << d << "\n";
    return 0;
}
