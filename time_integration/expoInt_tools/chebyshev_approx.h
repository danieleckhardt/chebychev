#ifndef DATA_TYPES_CHEBYSHEV_APPROX_H_
#define DATA_TYPES_CHEBYSHEV_APPROX_H_

#include <complex>
#include <cmath>
#include <functional>
#include <variant>
#include <fftw3.h>

#include "../../types/Ellipse.h"
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h> 
#include <deal.II/lac/sparse_matrix.h>


using namespace std::complex_literals; // needs c++14
using namespace dealii;
using BMatrixVariant = std::variant<
    std::function<int(FullMatrix<std::complex<double>>&)>,
    std::function<int(Vector<double>&, Vector<double>&)>,
    std::function<int(Vector<std::complex<double>>&, Vector<std::complex<double>>&)>
>;

namespace MavesS{
namespace TimeIntegration
{   

template<int dim>
class Chebyshev{
    /*
    Chebyshev class for Chebyshev approximation of functions on an ellipse.
    */

    public:
        Chebyshev(const int degree, Types::Ellipse<dim>& ellipse, double& time_step_size, const SparseMatrix<double>& A, const SparseMatrix<double>& M );
        void get_degree(int& degree_) const;
        void get_coefficients(std::vector<double>& coeffs) const;
        void set_chebyshev_tn_basis(int degree_);
        void set_degree(int degree_);
        void coefficient_fft(std::function<std::complex<double>(std::complex<double>)> func,std::function<std::complex<double>(std::complex<double>)> transform_func);
        void estimate_degree_ellipse_by_error(std::function<std::complex<double>(std::complex<double>)> f, double tol = 1e-2, int max_iter = 30);
        void estimate_degree_ellipse_by_estimate(int phi_k, double tol = 1e-2);
        Types::Ellipse<dim>& ellipse;
        std::complex<double> scalar_run(std::function<std::complex<double>(std::complex<double>)> func, std::complex<double> x);
        std::complex<double> scalar_run( std::complex<double> x );
        void matrix_run( BMatrixVariant Matrix_mult, std::function<std::complex<double>(std::complex<double>)> func, FullMatrix<std::complex<double>> & blockvector, const std::optional<std::complex<double>> &ref_point_opt, double tol = 1., int k_phi = 1 );
        void get_count_mv(int& mv_count_out) const { mv_count_out = mv_count; }
    private:
        double scalar_product(const Vector<std::complex<double>>& a,
                                        const Vector<std::complex<double>>& b,
                                        const SparseMatrix<double>& Matrix,
                                        bool use_matrix = true ) ;
        double calculate_residual(int k_phi);
        void apply_scaled_chebyshev( BMatrixVariant Matrix_mult,
                                     const std::complex<double> &factor,
                                     double &center,
                                     const std::complex<double> &inv_scale,
                                     std::complex<double> &scaling_ref,
                                     std::complex<double> &scaling_new,
                                     std::complex<double> &scaling_old,
                              FullMatrix<std::complex<double>> &blockvector);
        void apply_scaled_chebyshev_alternativ(
                                    BMatrixVariant Matrix_mult,
                                    double &center,
                                    double &radiusx,
                                    double &radiusy,
                                    FullMatrix<std::complex<double>> &blockvector); 
        void apply_residual_chebyshev( BMatrixVariant Matrix_mult,
                                       const std::complex<double> &factor,
                                       double &center,
                                       double &radiusx,
                                       double &radiusy,
                                       const std::complex<double> &inv_scale,
                                       FullMatrix<std::complex<double>> &blockvector,
                                       double tol,
                                       int k_phi);
        int degree;
        int mv_count = 0;
        double& time_step_size;
        double tol;
        std::vector<double> coefficients;
        FullMatrix<std::complex<double>> cheb_n_3, cheb_n_2, cheb_n_1, cheb_n, cheb_np1;
        FullMatrix<std::complex<double>> Matrix_x_vector,vector_primes;        
        FullMatrix<std::complex<double>> res;    
        const SparseMatrix<double>& A;
        const SparseMatrix<double>& M;    
};

}/* namespace TimeIntegration*/
}/* namespace MavesS */

#endif // DATA_TYPES_CHEBYsHEV_APPROX_H_