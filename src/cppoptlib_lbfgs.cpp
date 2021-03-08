#include "../inc/cppoptlib/meta.h"
#include "../inc/cppoptlib/problem.h"
#include "../inc/cppoptlib/solver/bfgssolver.h"
#include "../inc/cppoptlib/solver/lbfgssolver.h"

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

template <class T> class SoftDTWCost : public Problem<T>
{
  private:
    const T *X;
    const uint m;
    const uint n;
    const uint k;
    const T gamma;
    T *G;
    T *Z;

  public:
    using typename Problem<T>::TVector;
    SoftDTWCost(const T *X_, const uint m_, const uint n_, const uint k_,
                const T gamma_)
        : X(X_), m(m_), n(n_), k(k_), gamma(gamma_)
    {
        G = new T[m * k]{0};
        Z = new T[m * k]{0};
    }
    T value(const TVector &x)
    {
        memset(G, 0, m * k * sizeof(T));
        T fx = 0.0;
        // copy x to Z
        //        std::cout << "z:\n";
        for (uint i = 0; i < m; i++)
        {
            // TODO make this multidimensional
            //          std::cout << x(i) << " ";
            Z[i] = x(i);
        }

        // compute softdtw barycenter cost
        fx = barycenter_cost<T>(Z, X, G, m, n, k, gamma);
        memset(Z, 0.0, m * k * sizeof(T));
        return fx;
    }
    void gradient(const TVector &x, TVector &grad)
    {
        T fx = 0.0;
        // copy x to Z and grad to G
        std::cout << "z:\n";
        for (uint i = 0; i < m; i++)
        {
            // TODO make this multidimensional
            std::cout << x(i) << " ";
            Z[i] = x(i);
            G[i] = grad(i);
        }

        // compute softdtw barycenter cost
        fx = barycenter_cost<T>(Z, X, G, m, n, k, gamma);
        std::cout << "\ncost: " << fx << "\n";

        // copy G back to grad
        for (uint i = 0; i < m; i++)
        {
            grad(i) = G[i];
        }
        // Zero out G and Z
        memset(G, 0, m * k * sizeof(T));
        memset(Z, 0.0, m * k * sizeof(T));
    }
    ~SoftDTWCost()
    {
        delete[] G;
        delete[] Z;
    }
};

int main()
{
    const int m = 96;
    const int n = 10;
    const int k = 1;
    const float gamma = 0.1;
    // test data
    float X[m * n]{0};
    test_load_X(X);
    float Z[m]{0};
    test_load_Z(Z);

    SoftDTWCost<float> f((float *)&X, m, n, k, gamma);
    BfgsSolver<SoftDTWCost<float>> solver;

    // Initial guess
    VectorXf x = VectorXf::Zero(m);
    // copy Z to x
    for (uint i = 0; i < m; i++)
    {
        x(i) = Z[i];
    }
    solver.minimize(f, x);

    std::cout << "x = \n" << x.transpose() << std::endl;
    std::cout << "f(x) = " << f(x) << std::endl;

    return 0;
}
