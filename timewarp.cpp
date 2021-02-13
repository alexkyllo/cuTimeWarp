/**
 * @file timewarp.cpp
 */
#include <iostream>
#include <limits>
/**
 * @param a The first series array
 * @param b The second series array
 * @param w An m+1 x n+1 array that will be filled with the alignment values.
 * @param m Length of array a
 * @param n Length of array b
 */
template <class T> T timewarp(T *a, T *b, T *w, int m, int n)
{
    // Create an m*n matrix for the warp path
    // and initialize it to infinite distances
    m++;
    n++;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // this doesn't work for int type. Only float and double
            w[i * n + j] = std::numeric_limits<T>::infinity();
        }
    }
    w[0] = 0.0;

    // Iterate over each cell of the matrix to compute
    // lowest cost path through the preceding neighbor cells
    for (int i = 1; i < m; i++)
    {
        for (int j = 1; j < n; j++)
        {
            T cost = std::abs(a[i-1] - b[j-1]);
            double prev_min =
                std::min(w[(i - 1) * n + j],
                         std::min(w[i * n + j - 1], w[(i - 1) * n + j - 1]));
            w[i * n + j] = cost + prev_min;
        }
    }
    // Return the total cost of the warp path
    T path_cost = w[m*n-1];
    return path_cost;
}

int main()
{
    // test values
    int m = 5;
    int n = 8;
    float *a = new float[m]{1.0, 2.0, 3.0, 3.0, 5.0};
    float *b = new float[n]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};
    //
    
    float *w = new float[(m+1)*(n+1)]{0.0};
    float cost = timewarp<float>(a, b, w, m, n);
    // Should print: 3
    std::cout << "DTW distance: " << cost << "\n";
    std::cout << "Alignment matrix:\n";
    for (int i = 0; i < m+1; i++) {
        for (int j = 0; j < n+1; j++) {
            std::cout << w[i * (n+1) + j] << " ";
        }
        std::cout << "\n";
    }
    delete[] a;
    delete[] b;
    delete[] w;
    return 0;
}
