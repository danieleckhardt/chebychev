#ifndef MATRIX_FUNCTION_H_
#define MATRIX_FUNCTION_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>


// #include <deal.II/dofs/dof_tools.h>


#include "../data/ExactSolution.h"
#include "../data/InitialConditions.h"
#include "../types/BlockMatrix.h"
#include "../types/Functions_for_matrix.h"

#include "../generating/MicroProblem.h"


#include "../time_integration/expoInt_tools/expoInt.h"
#include "../tools/Output.h"
#include "global.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/foreach.hpp>

#include <iostream>
#include <fstream>
#include <random>
#include <lapacke.h>
#include <complex>
#include <functional>
#include <filesystem>
#include <chrono>


namespace MavesS{
    using namespace dealii;
    using namespace TimeIntegration;
    using namespace Microproblem;
    using BMatrixVariant = std::variant<
    std::function<int(FullMatrix<std::complex<double>>&)>,
    std::function<int(Vector<double>&, Vector<double>&)>,
    std::function<int(Vector<std::complex<double>>&, Vector<std::complex<double>>&)>
    >;

    template<int dim>
    class MatrixFunction{
        public:
            enum MethodMatrixFunction
            {
                poly_krylov,
                rat_krylov,
                cheb
            };
            MatrixFunction();
            void run(std::string beta);
            void set_method (const MethodMatrixFunction method_in ){this->method = method_in;}
            void setup_system();
            void create_ref_matrix_func();
            void create_ref_ellipse();
            void create_ref_matrix_func_ode();
            void create_ref_matrix_func_poly_krylov();
            void setup_matrices( const double coef_damping);
            void get_results(double& err_out, double& computation_time_mf_out, int& mv_count_total_out){
                err_out = err;
                computation_time_mf_out = computation_time_mf;
                mv_count_total_out = mv_count_total;
            };
            void get_size(int& size_out ){size_out = size;};
            ExpoInt<dim> expoint;

            private:
                double error_calc_ell(double& ref_radiusx, double& ref_radiusy, double& ref_center, std::complex<double>& ref_eccentricity);
                double error_calc_matrix_function(const Vector<double>& ref_vec_long, Vector<double>& calc_vec1, Vector<double>& calc_vec2);
                // void    matrix_vector_product(Vector<double>& a, Vector<double>& b );
                double scalar_product(const Vector<double>& a,
                                                const Vector<double>& b,
                                                const SparseMatrix<double>& matrix,
                                                bool use_matrix = true) ;
                MethodMatrixFunction method;
                Triangulation<dim> triangulation;
                
                DoFHandler<dim>    dof_handler;
                FE_Q<dim>          fe; 
                SparsityPattern    sparsity_pattern;

                Vector<double> b_vec1; 
                Vector<double> b_vec2;
                Vector<double> ref_vec_in;
                Vector<double> ref_vec_out;

                SparseMatrix<double>             laplace_matrix;
                SparseMatrix<double>             homogeneous_matrix;
                SparseMatrix<double>             damping_matrix;
                SparseMatrix<double>             mass_matrix;
                FullMatrix<std::complex<double>> eigenvectors_ref;
                FullMatrix<std::complex<double>> eigenvectors_inverse_ref;
                FullMatrix<double>               ref_matrix_func;
                FullMatrix<double>               identity_m;

                AffineConstraints<double> constraints;

                // setup blockmatrix
                std::vector<int> first_index_set;
                std::vector<int> second_index_set;
                //
                std::vector<std::complex<double>> eigenvalues;
                FullMatrix<std::complex<double>> eigenvectors;


                double tau;
                double coef_a  ;
                double coef_damping ;
                double epsilon;
                std::vector<double>  tol_vec;
                double tol_ell;
                double err;
                std::vector<double> err_vec;
                double err_ell; 
                double computation_time_mf;
                std::vector<double> computation_time_mf_vec;
                double computation_time_ell;
                double rat_shift;
                double ref_radiusx, ref_radiusy, ref_center;
                std::complex<double> ref_eccentricity;
                int    size;
                int    degree;
                int phi_k;
                int mv_count_total;
                std::vector<int> mv_count_total_vec;

                BMatrixVariant bmatrix_vmult;
                BMatrixVariant bmatrix_vmult_inv;
                Types::Functions_for_Matrix<dim> func_matrix;
                std::function<std::complex<double>(std::complex<double>)> func_lambda;

                std::unique_ptr<Types::BlockMatrix<dim>> bmatrix;
                Output<dim,FullMatrix<double>> output_class;
                Output<dim,FullMatrix<std::complex<double>>> output_class_complex;
                Calc_eigen<dim> calc_eigen;
                TimerOutput     computing_timer;
                GeneratingHomCoeff<dim>   genHomCoeff{};


                std::filesystem::path currentPath = std::filesystem::current_path();


    };



};


#endif /* MATRIX_FUNCTION_H_ */