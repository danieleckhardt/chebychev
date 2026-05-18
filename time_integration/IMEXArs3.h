#ifndef IMEXARS_H_ 
#define IMEXARS_H_

#include <math.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include "../data/RHS.h"  
#include "../data/Coefficients.h"
#include "../data/Boundary.h"
#include "../data/InitialConditions.h"
#include "../processing/solving.h"
#include "TimeIntegrator.h"

using namespace dealii;

namespace MavesS //Multi-Waves-Scale
{
using namespace Data;
using namespace Solving;
namespace TimeIntegration
{
    template<int dim>
    class Imexars3 : public TimeIntegrator<dim>
    {
        public:
            Imexars3(FE_Q<dim>&  fe_macro, DoFHandler<dim>&  dof_handler_macro,AffineConstraints<double>& constraints_macro, 
                                    SparseMatrix<double>& homogeneous_matrix,SparseMatrix<double>& damping_matrix,SparseMatrix<double>& mass_matrix,
                                    const double time_step_size, const double & degree_nl_v, const int example, const bool damped, const bool nonlinear);
            void integrate_step(Vector<double>& solution_macro_u,Vector<double>& solution_macro_v,
                                                const Vector<double>& old_solution_macro_u, const Vector<double>& old_solution_macro_v, const double time_macro) override;
            void setup_system_macro_space() override;
            void setup_system_macro_time( double new_timestep) override;
            void assemble_system_V(const double time_macro){} ;

        protected: 
            using TimeIntegrator<dim>::dof_handler_macro;
            using TimeIntegrator<dim>::fe_macro;
            using TimeIntegrator<dim>::constraints_macro;
            using TimeIntegrator<dim>::degree_nl_v;
            using TimeIntegrator<dim>::tol;


        private:
            //
            const bool damped;
            const bool nonlinear;
            bool   first_iteration  = true;
            //
            double time_step_size;
            double theta = (3 + sqrt(3))/ 6.;
            double initial_rhs_norm{};
            //
            RightHandSide<dim> rhs_function;
            //
            Vector<double> forcing_terms;
            Vector<double> pre_system_rhs_macro;
            Vector<double> solution_macro_u_first_stage;
            Vector<double> solution_macro_u_second_stage;
            Vector<double> solution_macro_v_first_stage;
            Vector<double> solution_macro_v_second_stage;
            Vector<double> tmp; 
            Vector<double> system_rhs_macro;
            //
            SparseMatrix<double>& mass_matrix;
            SparseMatrix<double>& homogeneous_matrix; // this matrix will be created by calculate the micro solution
            SparseMatrix<double>& damping_matrix;  
            SparseMatrix<double> system_matrix_macro;
            //                        
            Solver<dim> solver;
            
            void assemble_system_V_first_stage(const Vector<double>  &old_solution_macro_u, const Vector<double>& old_solution_macro_v, const double &time_macro);
            void assemble_system_V( const Vector<double>  &old_solution_macro_u,Vector<double>  &solution_macro_u_first_stage,Vector<double>  &solution_macro_u_second_stage,const Vector<double>  &old_solution_macro_v, Vector<double>& solution_macro_v_first_stage,Vector<double>& solution_macro_v_second_stage, const double &time_macro);
            void assemble_system_V_second_stage( const Vector<double>  &old_solution_macro_u, const Vector<double>& old_solution_macro_v, Vector<double>& solution_macro_u_first_stage, Vector<double>& solution_macro_v_first_stage,const double &time_macro);
            void solve_U_stage(Vector<double>& solution_macro_u, const Vector<double>& old_solution_macro_u,Vector<double>& solution_macro_v_first_stage, Vector<double>& solution_macro_v_second_stage, const double time_macro_step_first_factor, const double time_macro_step_second_factor);
    };
}
}
 #endif /* IMEXARS_H_ */