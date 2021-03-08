#include "../inc/Eigen/Core"
#include "../inc/LBFGS.h"
#include "soft_dtw_cpu.hpp"
#include <cstring>

using Eigen::VectorXf;
using namespace LBFGSpp;

template <class T> class SoftDTWCost
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
    SoftDTWCost(const T *X_, const uint m_, const uint n_, const uint k_,
                const T gamma_)
        : X(X_), m(m_), n(n_), k(k_), gamma(gamma_)
    {
        G = new T[m * k]{0};
        Z = new T[m * k]{0};
    }
    T operator()(const VectorXf &x, VectorXf &grad)
    {
        T fx = 0.0;
        // copy x to Z and grad to G
        std::cout << "z:\n";
        for (int i = 0; i < m; i++)
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
        for (int i = 0; i < m; i++)
        {
            grad(i) = G[i];
        }
        // Zero out G and Z
        memset(G, 0, m * k * sizeof(T));
        memset(Z, 0.0, m * k * sizeof(T));
        return fx;
    }
};
