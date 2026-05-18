#ifndef EXPINT_H_
#define EXPINT_H_

#include <cmath>
#include <iomanip>
#include <functional> // Required for std::function
#include <fftw3.h>
#include <chrono>
#include <random>
#include <iostream>



#include "../../data/RHS.h"  
#include "../../data/Coefficients.h"
#include "../../data/Boundary.h"
#include "../../data/InitialConditions.h"
#include "../../processing/solving.h"
#include "../../types/InverseMatrix.h"
#include "../../types/BlockMatrix.h"
#include "../../types/Ellipse.h"
#include "../TimeIntegrator.h"
#include "chebyshev_approx.h"
#include "calc_eigen.h"
#include "arnoldikrylov.h"
#include "findellipse.h"

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/lapack_full_matrix.h>

// #include <lapacke.h> // LAPACK
// #include <Spectra/GenEigsSolver.h>
// #include <Eigen/Core>

using namespace dealii;
using namespace std::complex_literals; // needs c++14
using BMatrixVariant = std::variant<
    std::function<int(FullMatrix<std::complex<double>>&)>,
    std::function<int(Vector<double>&, Vector<double>&)>,
    std::function<int(Vector<std::complex<double>>&, Vector<std::complex<double>>&)>
>;
using TypeVariant = std::variant<double, std::complex<double>>;

namespace MavesS //Multi-Waves-Scale
{
using namespace Data;
using namespace Solving;
namespace TimeIntegration
{

    template<int dim>
    class ExpoInt : public TimeIntegrator<dim>
    // This class compute  Matrixfunctions for a give n Matrix Vector multiplication object
    // use poly Krylov, shift and invert or the chebychev method
    {
        public:
            ExpoInt(FE_Q<dim>&  fe_macro, DoFHandler<dim>&  dof_handler_macro,AffineConstraints<double>& constraints_macro,  
                SparseMatrix<double>& homogeneous_matrix, SparseMatrix<double>& damping_matrix, SparseMatrix<double>& mass_matrix,
                double& time_step_size, const double & degree_nl_v, bool info_ellipse = false);

            void setup_system_macro_space() override;                                                
            void setup_system_macro_time( double new_timestep) override;
            void matrix_func_cheb(BMatrixVariant Matrix_mult, std::function<std::complex<double>(std::complex<double>)> func, Vector<double> &u, Vector<double> &v, double tol, int k_phi = 1); 
            bool matrix_func_poly_krylov(BMatrixVariant  Matrix_mult,std::function<std::complex<double>(std::complex<double>)> func, Vector<double> &u, Vector<double> &v, const double tol); // approximate func([ 0 M^-1 ] [M^-1 A  M^-1 B])[u  v]
            bool matrix_func_rat_krylov(BMatrixVariant Matrix_mult,BMatrixVariant Matrix_mult_inv,std::function<std::complex<double>(std::complex<double>)> func, Vector<double> &u, Vector<double> &v, const double tol, const double shift_parameter = 1.0);
            std::vector<int> find_opt_ellipse_prototype(const FullMatrix<double>& A =  FullMatrix<double>(0, 0)) override;
            std::vector<std::complex<double>> find_opt_ellipse(double tol, Types::BlockMatrix<dim>* block_matrix_ptr = nullptr) override;
            int estimate_degree_phi_k(int phi_k, double tol = 1e-2, int init_degee = 10) override;
            int estimate_degree(std::function<std::complex<double>(std::complex<double>)> func,  double tol = 1e-2, int init_degee = 10) override;
            void set_max_iteration_krylov(int max_iter) override;
            void get_mv_count_per_time_step(int& mv_count_){mv_count_ = mv_count_per_step;};
            void store_info_ellipse(std::string filename);

            //
            using TimeIntegrator<dim>::ellipse_opt;
            int degree;
        protected:

            // calculate e^{\tau S}[u v] using krylov with S = [0 I ; - A  -B]
            int calc_all_eig(const FullMatrix<double>& A, int const size );
            int calc_max_eig(Types::BlockMatrix<dim>& block_matrix, int const k, double const tol );
            using TimeIntegrator<dim>::fe_macro;
            using TimeIntegrator<dim>::constraints_macro;
            using TimeIntegrator<dim>::degree_nl_v;
            using TimeIntegrator<dim>::dof_handler_macro;
            bool info_ellipse;
            int mv_count_per_step = 0;
         //
            int max_iterations{};
            int iterations_needed;
            double& time_step_size;
            double norm_value_krylov = 1;
            double initial_rhs_norm{};
            double a_out,b_out,c_out;

        //
            Vector<double> tmp_vec;
            Vector<double> system_rhs_macro;
            Vector<double> res_rat_vec;

            std::vector<double>  tmp_h;
            std::vector<Vector<double>> krylov_vectors_u; 
            std::vector<Vector<double>> krylov_vectors_v;
            std::vector<double> matrix_funcxe1_krylov{};
            
            std::vector<std::complex<double>> eigenvalues;
            //
            SparseMatrix<double>& mass_matrix;
            SparseMatrix<double>& homogeneous_matrix; // this matrix will be created by calculate the micro solution
            SparseMatrix<double>& damping_matrix;  
            //
            std::vector<int> ellipse_config;
            FullMatrix<std::complex<double>> eigenvectors;
            FullMatrix<std::complex<double>> approx_result;
            std::unique_ptr<Types::InverseMatrix<dim,double>> M_inv;
            // std::unique_ptr<Types::InverseMatrix<dim,std::complex<double>>> M_inv_c;

        //
            Chebyshev<dim> cheb_approx;
            Calc_eigen<dim> calc_eigen;
            ArnoldiKrylov<dim, double> arnoldi_krylov;
            std::unique_ptr<FindEllipse<dim>> find_ellipse;
            Output< dim, FullMatrix<double> > output_class;
            Output< dim, FullMatrix<std::complex<double>> > output_class_complex;


    };
}
}
#endif /* EXPINT_H_ */
