#include "../src/soft_dtw_cpu.hpp"
#include "catch.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

bool is_close(float a, float b, float tol = 0.0001)
{
    return std::abs(a - b) < tol;
}

TEST_CASE("CBLAS_SGEMM")
{
    int m = 4;
    int k = 2;
    int n = 3;
    float *X = new float[m * k]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float *Y = new float[k * n]{1.5, 2.6, 3.7, 4.8, 5.9, 6.1};
    float *XY = new float[m * n]{0};
    cblas_sgemm(CblasRowMajor, // Row-major striding
                CblasNoTrans,  // Do not transpose A
                CblasTrans,    // Transpose B
                m,             // Rows in A
                n,             // Columns in B
                k,             // Columns in A
                2.0,           // Alpha (scalar used to scale A*B)
                X,             // Input Matrix A
                k,             // LDA stride of matrix A
                Y,             // Input Matrix B
                k,             // LDA stride of matrix B
                0.0,           // Beta (scalar used to scale matrix C)
                XY,            // Result Matrix C
                n);            // LDA stride of matrix C
    float *XYE = new float[12]{13.4, 26.6, 36.2,  29.8, 60.6,  84.2,
                               46.2, 94.6, 132.2, 62.6, 128.6, 180.2};
    for (int i = 0; i < (m * n); i++)
    {
        REQUIRE(is_close(XY[i], XYE[i]));
    }
    delete[] X;
    delete[] Y;
    delete[] XY;
    delete[] XYE;
}

TEST_CASE("test squared euclidean distance 2d")
{
    int m = 4;
    int k = 2;
    int n = 3;
    float *X = new float[m * k]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float *Y = new float[k * n]{1.5, 2.6, 3.7, 4.8, 5.9, 6.1};
    float *D = new float[m * n]{0};
    sq_euclidean_distance<float>(X, Y, D, m, n, k);
    /**
       sklearn.metrics.pairwise.euclidean_distances(X, Y, squared=True)

       array([[ 0.61, 15.13, 40.82],
              [ 4.21,  1.13, 12.82],
              [23.81,  3.13,  0.82],
              [59.41, 21.13,  4.82]])
     */

    REQUIRE(is_close(D[0], 0.61));
    REQUIRE(is_close(D[1], 15.13));
    REQUIRE(is_close(D[2], 40.82));
    REQUIRE(is_close(D[3], 4.21));
    REQUIRE(is_close(D[4], 1.13));
    REQUIRE(is_close(D[5], 12.82));
    REQUIRE(is_close(D[6], 23.81));
    REQUIRE(is_close(D[7], 3.13));
    REQUIRE(is_close(D[8], 0.82));
    REQUIRE(is_close(D[9], 59.41));
    REQUIRE(is_close(D[10], 21.13));
    REQUIRE(is_close(D[11], 4.82));
    delete[] X;
    delete[] Y;
    delete[] D;
}

TEST_CASE("test squared euclidean distance 1d")
{
    int m = 5;
    int k = 1;
    int n = 8;
    float *a = new float[m]{1.0, 2.0, 3.0, 3.0, 5.0};
    float *b = new float[n]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};
    float *D = new float[m * n]{0};
    sq_euclidean_distance<float>(a, b, D, m, n, k);
    /**
       sklearn.metrics.pairwise.euclidean_distances(X, Y, squared=True)

       array([[ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  9.],
              [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  4.],
              [ 4.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
              [ 4.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
              [16.,  9.,  9.,  9.,  9.,  9.,  9.,  1.]])
    */

    REQUIRE(is_close(D[0], 0.0));
    REQUIRE(is_close(D[1], 1.0));
    REQUIRE(is_close(D[7], 9.0));
    REQUIRE(is_close(D[8], 1.0));
    REQUIRE(is_close(D[15], 4.0));
    REQUIRE(is_close(D[16], 4.0));
    REQUIRE(is_close(D[24], 4.0));
    REQUIRE(is_close(D[32], 16.0));
    REQUIRE(is_close(D[33], 9.0));
    REQUIRE(is_close(D[39], 1.0));

    delete[] a;
    delete[] b;
    delete[] D;
}

TEST_CASE("soft dtw")
{
    int m = 5;
    int n = 8;
    float gamma = 0.1;
    float *a = new float[m]{1.0, 2.0, 3.0, 3.0, 5.0};
    float *b = new float[n]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};
    float *R = new float[(m + 1) * (n + 1)]{0.0};
    float cost = softdtw<float>(a, b, R, m, n, gamma);
    REQUIRE(is_close(2.80539, cost));
    delete[] a;
    delete[] b;
    delete[] R;
}

TEST_CASE("soft dtw for distance matrix (1d ts)")
{
    int m = 5;
    int k = 1;
    int n = 8;
    float gamma = 0.1;
    float *a = new float[m]{1.0, 2.0, 3.0, 3.0, 5.0};
    float *b = new float[n]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};
    float *D = new float[m * n]{0};
    sq_euclidean_distance<float>(a, b, D, m, n, k);

    float *R = new float[(m + 2) * (n + 2)]{0.0};
    float cost = softdtw<float>(D, R, m, n, gamma);
    /*
R expected:
[0. inf     inf     inf     inf     inf     inf     inf       inf          inf]
[inf  0.      1.      2.      3.      4.      5.      6.       15.         inf]
[inf  1.     -0.     -0.     -0.     -0.     -0.     -0.        4.         inf]
[inf  5.      1.      0.9307  0.9307  0.9307  0.9307  0.9307    1.         inf]
[inf  9.      2.      1.8901  1.8614  1.8613  1.8613  1.8613    1.8901     inf]
[inf 25.     11.     10.8614 10.8054 10.792  10.792  10.792     2.8054     inf]
[inf inf     inf     inf     inf     inf     inf     inf           inf     inf]
    */
    REQUIRE(is_close(2.80539, cost));
    delete[] a;
    delete[] b;
    delete[] D;
    delete[] R;
}

TEST_CASE("soft dtw gradient")
{
    int m = 5;
    int n = 8;
    double gamma = 0.1;
    double *a = new double[m]{1.0, 2.0, 3.0, 3.0, 5.0};
    double *b = new double[n]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};
    double *D = new double[m * n]{0.0};
    double *R = new double[(m + 2) * (n + 2)]{0.0};
    double *E = new double[m * n]{0.0};
    sq_euclidean_distance(a, b, D, m, n, 1);
    softdtw<double>(D, R, m, n, gamma);
    softdtw_grad<double>(D, R, E, m, n, gamma);
    // for (int i = 0; i < m; i++)
    // {
    //     for (int j = 0; j < n; j++)
    //     {
    //         std::cout << E[i * n + j] << " ";
    //     }
    //     std::cout << "\n";
    // }
    REQUIRE(is_close(E[0], 1.0));
    REQUIRE(is_close(E[1], 0.0001));
    REQUIRE(is_close(E[13], 0.8571));
    REQUIRE(is_close(E[14], 0.4285));
    REQUIRE(is_close(E[21], 0.2857));
    REQUIRE(is_close(E[22], 0.5714));
    REQUIRE(is_close(E[23], 0.1429));
    REQUIRE(is_close(E[31], 0.4286));
    REQUIRE(is_close(E[39], 1.0));
    /* Expected gradient:
array([[[1.    , 0.0001, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
        [0.    , 1.    , 1.    , 1.    , 1.    , 0.8571, 0.4285, 0.    ],
        [0.    , 0.    , 0.    , 0.    , 0.    , 0.2857, 0.5714, 0.1429],
        [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.5714, 0.4286],
        [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.    ]]])
     */
    delete[] a;
    delete[] b;
    delete[] D;
    delete[] R;
    delete[] E;
}

TEST_CASE("test jacobian product")
{
    const uint m = 96;
    const float gamma = 0.1;
    float X[m] = {0};
    float Z[m] = {0};
    float G[m] = {0};  // expected
    float G0[m] = {0}; // actual
    float D[m * m] = {0};
    float R[(m + 2) * (m + 2)] = {0};
    float E[m * m] = {0};
    std::ifstream fileX("test/test_jacobian_X0.txt");
    std::ifstream fileZ("test/test_jacobian_Z.txt");
    std::ifstream fileG("test/test_jacobian.txt");
    std::string buffer;
    std::stringstream ss;
    float temp;
    for (uint i = 0; i < m; i++)
    {
        getline(fileX, buffer);
        ss.str(buffer);
        ss >> temp;
        X[i] = temp;
        ss.clear();

        getline(fileZ, buffer);
        ss.str(buffer);
        ss >> temp;
        Z[i] = temp;
        ss.clear();

        getline(fileG, buffer);
        ss.str(buffer);
        ss >> temp;
        G[i] = temp;
        ss.clear();
    }
    sq_euclidean_distance<float>((float *)&Z, (float *)&X, (float *)&D, m, m,
                                 1);
    softdtw<float>((float *)&D, (float *)&R, m, m, gamma);
    softdtw_grad<float>((float *)&D, (float *)&R, (float *)&E, m, m, gamma);
    jacobian_prod_sq_euc((float *)&Z, (float *)&X, (float *)&E, G0, m, m, 1);
    for (uint i = 0; i < m; i++)
    {
        REQUIRE(is_close(G[i], G0[i], 0.001));
    }
}

TEST_CASE("test jacobian product 1")
{
    const uint m = 96;
    const uint n = 10;
    const float gamma = 0.1;
    float X[m * n] = {0};
    float Z[m] = {0};
    float G[m] = {0};  // expected
    float G0[m] = {0}; // actual

    std::ifstream fileX("test/test_ecg200_10.txt");
    std::ifstream fileZ("test/test_jacobian_Z.txt");
    std::ifstream fileG("test/test_jacobian_G0.txt");
    std::string buffer;
    std::stringstream ss;
    float temp;
    for (uint i = 0; i < n; i++)
    {
        getline(fileX, buffer);
        ss.str(buffer);
        for (uint j = 0; j < m; j++)
        {
            ss >> temp;
            X[i * m + j] = temp;
            ss.clear();
        }
    }
    for (uint i = 0; i < m; i++)
    {
        getline(fileZ, buffer);
        ss.str(buffer);
        ss >> temp;
        Z[i] = temp;
        ss.clear();

        getline(fileG, buffer);
        ss.str(buffer);
        ss >> temp;
        G[i] = temp;
        ss.clear();
    }
    // float D[m * m] = {0};
    // float R[(m + 2) * (m + 2)] = {0};
    // float E[m * m] = {0};
    // sq_euclidean_distance<float>((float *)&Z, (float *)&X, (float *)&D, m, m,
    //                              1);
    // softdtw<float>((float *)&D, (float *)&R, m, m, gamma);
    // softdtw_grad<float>((float *)&D, (float *)&R, (float *)&E, m, m, gamma);
    // jacobian_prod_sq_euc((float *)&Z, (float *)&X, (float *)&E, G0, m, m, 1);
    barycenter_cost<float>((float *)&Z, (float *)&X, (float *)&G0, m, n, 1,
                           gamma);
    // print_matrix((float *)&G0, m, 1);
    for (uint i = 0; i < m; i++)
    {
        REQUIRE(is_close(G[i], G0[i], 0.001));
    }
}

TEST_CASE("test barycenter cost")
{
    std::ifstream datafile("test/test_ecg200_10.txt");
    std::ifstream fileZ("test/test_jacobian_Z.txt");
    std::ifstream fileG("test/test_jacobian_G0.txt");
    // example data is 10 arrays of length 96
    const uint m = 96;
    const uint n = 10;
    float G0[m] = {0}; // actual
    float G[m] = {0};  // expected
    float X[m * n]{0};
    std::stringstream ss;
    std::string buffer;
    float temp;

    assert(datafile.is_open());

    for (uint i = 0; i < n && !datafile.eof(); i++)
    {
        getline(datafile, buffer);
        ss.str(buffer);
        for (uint j = 0; j < m; j++)
        {
            ss >> temp;
            X[i * m + j] = temp;
        }
        ss.clear();
    }
    float Z[m]{0};
    for (uint i = 0; i < m; i++)
    {
        getline(fileZ, buffer);
        ss.str(buffer);
        ss >> temp;
        Z[i] = temp;
        ss.clear();

        getline(fileG, buffer);
        ss.str(buffer);
        ss >> temp;
        G[i] = temp;
        ss.clear();
    }
    const float gamma = 0.1;
    float cost = barycenter_cost<float>((float *)&Z, (float *)&X, (float *)&G0,
                                        m, n, 1, gamma);
    REQUIRE(is_close(cost, 37.4417));
    // print_matrix((float *)&G, m, 1);
    // std::cout << "expected: \n";
    // print_matrix((float *)&G1, m, 1);
    for (uint i = 0; i < m; i++)
    {
        REQUIRE(is_close(G[i], G0[i], 0.001));
    }
}

TEST_CASE("test barycenter cost 1")
{
    std::ifstream datafile("test/test_ecg200_10.txt");
    std::ifstream fileZ("test/test_jacobian_Z1.txt");
    std::ifstream fileG("test/test_jacobian_G1.txt");
    // example data is 10 arrays of length 96
    const uint m = 96;
    const uint n = 10;
    float G[m] = {0};  // expected
    float G0[m] = {0}; // actual
    float X[m * n]{0};
    std::stringstream ss;
    std::string buffer;
    float temp;

    assert(datafile.is_open());

    for (uint i = 0; i < n && !datafile.eof(); i++)
    {
        getline(datafile, buffer);
        ss.str(buffer);
        for (uint j = 0; j < m; j++)
        {
            ss >> temp;
            X[m * i + j] = temp;
        }
        ss.clear();
    }
    float Z[m]{0};
    for (uint i = 0; i < m; i++)
    {
        getline(fileZ, buffer);
        ss.str(buffer);
        ss >> temp;
        Z[i] = temp;
        ss.clear();
        getline(fileG, buffer);
        ss.str(buffer);
        ss >> temp;
        G[i] = temp;
        ss.clear();
    }
    const float gamma = 0.1;
    float cost = barycenter_cost<float>((float *)&Z, (float *)&X, (float *)&G0,
                                        m, n, 1, gamma);
    REQUIRE(is_close(cost, -1.672, 0.001));
    for (uint i = 0; i < m; i++)
    {
        REQUIRE(is_close(G[i], G0[i], 0.01));
    }
}

// TEST_CASE("soft dtw barycenter")
// {
//     // read in the example time series line by line into float array
//     std::ifstream datafile("test/test_ecg200_10.txt");
//     // example data is 10 arrays of length 96
//     const uint m = 96;
//     const uint n = 10;
//     float X[m * n]{0};
//     std::stringstream ss;
//     std::string buffer;
//     float temp;

//     assert(datafile.is_open());

//     for (uint i = 0; i < n && !datafile.eof(); i++)
//     {
//         getline(datafile, buffer);
//         // std::cout << buffer << "\n";
//         ss.str(buffer);
//         for (uint j = 0; j < m; j++)
//         {
//             ss >> temp;
//             X[m * i + j] = temp;
//             // std::cout << X[m * i + j] << " ";
//         }
//         ss.clear();
//     }
//     float Z[m]{0};
//     const float gamma = 0.1;
//     const float tol = 0.0001;
//     const float max_iter = 100;
//     const float lr = 0.01;
//     float cost = find_softdtw_barycenter<float>((float *)&Z, (float *)&X, m,
//     1,
//                                                 n, gamma, tol, max_iter, lr);
//     // print the barycenter values
//     for (uint i = 0; i < m; i++)
//     {
//         std::cout << Z[i] << " ";
//     }
//     std::cout << std::endl;
//     std::cout << "cost: " << cost << "\n";
// }
