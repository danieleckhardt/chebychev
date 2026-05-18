#ifndef CALC_EIGEN_H_
#define CALC_EIGEN_H_

#define LAPACK_COMPLEX_CPP   

#include <lapacke.h>         

#include <cmath>
#include <iomanip>
#include <functional>

#include "../../types/InverseMatrix.h"
#include "../../types/BlockMatrix.h"
#include "../../types/BlockMatrixOp.h"
#include "../../types/BlockMatrixpetsc.h"

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <slepceps.h>
#include <Spectra/GenEigsSolver.h>
#include <Eigen/Core>

using namespace dealii;
using namespace std::complex_literals; // needs c++14
struct EigPair {
    std::complex<double> value;
    std::vector<std::complex<double>> vector;
};

namespace MavesS //Multi-Waves-Scale
{
namespace TimeIntegration
{

    template<int dim>
    class Calc_eigen 
    {
        public:
            Calc_eigen( std::vector<std::complex<double>>& eigenvalues, FullMatrix<std::complex<double>>& eigenvectors);
            int calc_all_eig(const FullMatrix<double>& A, int const size ); // This function calculates all eigenvalues and eigenvectors of a real matrix A using LAPACK's dgeev function.
            int calc_all_eig(const FullMatrix<std::complex<double>>& A, int const size ); // This function calculates all eigenvalues and eigenvectors of a complex matrix A using LAPACK's zgeev function.
            int calc_max_eig(Types::BlockMatrix<dim>& block_matrix, int const k, double const tol, std::string method = "spectra");
        private:   
            int calc_max_eig_spectra(Types::BlockMatrix<dim>& block_matrix, int const k, double const tol);
            int calc_max_eig_petsc(Types::BlockMatrix<dim>& block_matrix, int const k, double const tol);
            std::vector<std::complex<double>>& eigenvalues;
            //
            FullMatrix<std::complex<double>>& eigenvectors;
    };
}   
}
#endif /* CALC_EIGEN_H_ */
