#ifndef EXPOEULER_H_
#define EXPOEULER_H_

#include <cmath>
#include "../types/BlockMatrix.h"
#include "../types/Functions_for_matrix.h"
#include "../data/RHS.h"  
#include "../data/Coefficients.h"
#include "../data/Boundary.h"
#include "../data/InitialConditions.h"
#include "../processing/solving.h"
#include "expoInt_tools/expoInt.h"

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/lapack_full_matrix.h>

using namespace dealii;

namespace MavesS //Multi-Waves-Scale
{
using namespace Data;
using namespace Solving;
namespace TimeIntegration
{

    template<int dim>
    class ExpoEuler : public ExpoInt<dim>
    {
        public:
            ExpoEuler(FE_Q<dim>& fe_macro, DoFHandler<dim>& dof_handler_macro, AffineConstraints<double>& constraints_macro, 
                        SparseMatrix<double>& homogeneous_matrix, SparseMatrix<double>& damping_matrix, SparseMatrix<double>& mass_matrix, std::string method_calc_mfunc, 
                        double & time_step_size, const double & degree_nl_v, const int example, const bool damped, const bool nonlinear);

            void integrate_step(Vector<double>& solution_macro_u,Vector<double>& solution_macro_v, const Vector<double>&  old_solution_macro_u, const Vector<double>&  old_solution_macro_v, const double time_macro) override;
            void setup_system_macro_space() override;
            void setup_system_macro_time( double new_timestep) override;
            using ExpoInt<dim>::time_step_size;
            using TimeIntegrator<dim>::iteration_krylov;
            using TimeIntegrator<dim>::degree_cheb;
            using TimeIntegrator<dim>::mv_count;
            
        protected:
           
            using TimeIntegrator<dim>::dof_handler_macro;
            using TimeIntegrator<dim>::fe_macro;
            using TimeIntegrator<dim>::constraints_macro;
            using TimeIntegrator<dim>::degree_nl_v;
            using TimeIntegrator<dim>::tol;
            using ExpoInt<dim>::tmp_vec;
            using ExpoInt<dim>::damping_matrix;
            using ExpoInt<dim>::homogeneous_matrix;
            using ExpoInt<dim>::mass_matrix;
            using ExpoInt<dim>::M_inv;
            using ExpoInt<dim>::degree;
            using ExpoInt<dim>::arnoldi_krylov;
            using ExpoInt<dim>::cheb_approx;
            double rat_shift = 1.;
            
        private:
            const bool damped;
            const bool nonlinear;
            std::string method_calc_mfunc;
            
            RightHandSide<dim> rhs_function;

            Vector<double> forcing_terms;
            Vector<double> system_rhs_macro_u;
            Vector<double> system_rhs_macro_v;
            Output< dim, FullMatrix<double> > output_class;
            Types::Functions_for_Matrix<dim> func_matrix;
            BMatrixVariant bmatrix_vmult;
            BMatrixVariant bmatrix_vmult_inv;
            std::unique_ptr<Types::BlockMatrix<dim>> bmatrix;
            std::function<std::complex<double>(std::complex<double>)> phi1_lambda;
    };

}
}
#endif /* EXPOEULER_H_*/