#include "soft_dtw_cost.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

void test_load_X(float *X)
{
    std::ifstream datafile("test/test_ecg200_10.txt");
    // example data is 10 arrays of length 96
    const uint m = 96;
    const uint n = 10;
    std::stringstream ss;
    std::string buffer;
    float temp;

    assert(datafile.is_open());

    for (uint i = 0; i < n && !datafile.eof(); i++)
    {
        getline(datafile, buffer);
        // std::cout << buffer << "\n";
        ss.str(buffer);
        for (uint j = 0; j < m; j++)
        {
            ss >> temp;
            X[m * i + j] = temp;
            // std::cout << X[m * i + j] << " ";
        }
        ss.clear();
    }
}

void test_load_Z(float *Z)
{
    const uint m = 96;
    std::ifstream fileZ("test/test_jacobian_Z.txt");
    std::string buffer;
    std::stringstream ss;
    float temp;
    for (uint i = 0; i < m; i++)
    {
        getline(fileZ, buffer);
        ss.str(buffer);
        ss >> temp;
        Z[i] = temp;
        ss.clear();
    }
}

int main()
{
    const int m = 96;
    const int n = 10;
    const int k = 1;
    const float gamma = 0.1;
    // Set up parameters
    LBFGSParam<float> param;
    param.epsilon = 1e-3;
    param.max_iterations = 50;
    // param.max_linesearch = 20;
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;

    // test data
    float X[m * n]{0};
    test_load_X(X);
    float Z[m]{0};
    test_load_Z(Z);

    // Create solver and function object
    // LBFGSSolver<float> solver(param);
    // LBFGSSolver<float, LineSearchBacktracking> solver(param);
    LBFGSSolver<float, LineSearchNocedalWright> solver(param);
    // VectorXf lb = VectorXf::Constant(n, -3.0);
    // VectorXf ub = VectorXf::Constant(n, 3.0);
    SoftDTWCost<float> fun((float *)&X, m, n, k, gamma);

    // Initial guess
    VectorXf x = VectorXf::Zero(m);
    // copy Z to x
    for (uint i = 0; i < m; i++)
    {
        x(i) = Z[i];
    }
    // x will be overwritten to be the best point found
    float fx;
    int niter = solver.minimize(fun, x, fx);
    // int niter = solver.minimize(fun, x, fx, lb, ub);

    std::cout << niter << " iterations" << std::endl;
    std::cout << "x = \n" << x.transpose() << std::endl;
    std::cout << "f(x) = " << fx << std::endl;

    return 0;
}
