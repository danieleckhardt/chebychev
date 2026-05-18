#ifndef TimeIntegrator_H_ 
#define TimeIntegrator_H_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

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
#include "../tools/Output.h"
#include "../types/BlockMatrix.h"
#include "../types/Ellipse.h"

#include <cmath> // pow

using namespace dealii;

namespace MavesS //Multi-Waves-Scale
{
using namespace Data;
using namespace Solving;
namespace TimeIntegration
{
    template<int dim>
    class TimeIntegrator
    {
        public:
            //
            TimeIntegrator(FE_Q<dim>&  fe_macro, DoFHandler<dim>&  dof_handler_macro,AffineConstraints<double>& constraints_macro,  const double & degree_nl_v);
            virtual void integrate_step(Vector<double>& solution_macro_u,Vector<double>& solution_macro_v, const Vector<double>& old_solution_macro_u, const Vector<double>& old_solution_macro_v, const double time_macro){};
            virtual void setup_system_macro_space(){};
            virtual void setup_system_macro_time( double new_timestep){};
            // only for expoInt
            virtual void set_max_iteration_krylov( int max_iter){};
            virtual int estimate_degree(std::function<std::complex<double>(std::complex<double>)> func,  double tol,  int init_degree  ){ return init_degree; };
            virtual int estimate_degree_phi_k(int phi_k, double tol, int init_degree ){return init_degree;};
            virtual std::vector<std::complex<double>> find_opt_ellipse(double tol, Types::BlockMatrix<dim>* block_matrix_ptr = nullptr){ return {};};
            virtual std::vector<int> find_opt_ellipse_prototype(const FullMatrix<double>& A ){return {};};
            Types::Ellipse<dim> ellipse_opt;
            unsigned  n_iterations; // only needed for implicit methods
            int iteration_krylov; // only for expoInt with krylov
            int degree_cheb;           // only for cheb
            int mv_count;
            void set_tol(const double tol_){tol = tol_;};
            //
        protected:
            //
            double tol = 1e-8;
            FE_Q<dim> & fe_macro;  
            DoFHandler<dim>& dof_handler_macro;
            AffineConstraints<double>& constraints_macro;
            const double & degree_nl_v;
            void apply_Uboundary_condition( SparseMatrix<double> &system_matrix_macro,Vector<double> &system_rhs_macro, Vector<double> &solution_macro_u);  
            void apply_Vboundary_condition( SparseMatrix<double> &system_matrix_macro,Vector<double> &system_rhs_macro, Vector<double> &solution_macro_v);  
            void compute_nl_term(const Vector<double> &data, Vector<double> &nl_term);
    };
}
}

 #endif /* TimeIntegrator_H_ */