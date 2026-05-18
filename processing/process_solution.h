#ifndef PROCESS_SOLUTION_H_
#define PROCESS_SOLUTION_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
 
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include "../data/ExactSolution.h"


#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <limits>
#include <string>
#include <chrono> // Walltime



namespace MavesS
{

namespace processing{
using namespace dealii;
using namespace Data;
    template <int dim>
    class Process_Solutions
    {
        public:
            Process_Solutions(DoFHandler<dim>&  dof_handler_macro, DoFHandler<dim>&  dof_handler_reference, Triangulation<dim> & triangulation_macro,  Triangulation<dim> & triangulation_reference,TimerOutput & computing_timer );
            void save_referencesolution(const Vector<double> &solution,const double &time, const bool &derivative_u, const bool &nonlinear, const bool &damped,    std::string example, std::string coefficient_type, unsigned int level_ref, std::string currentPath ) const;
            void load_referencesolution(const double &time, const bool &derivative_u, const bool &nonlinear, const bool &damped, std::string example,   std::string coef_type,  unsigned int level_ref, Vector<double> & solution_ref, std::string currentPath  );
            template <typename parametertype>
            void process_exactsolution(Vector<double> &solution, Solution<dim> & exact_solution,  Vector<double> &solution_prime, Solution_prime<dim> & exact_solution_prime, ConvergenceTable &  convergence_table_L2, ConvergenceTable &  convergence_table_H1, ConvergenceTable &  convergence_table_total,  const unsigned int fe_degree, std::string & convergence_parameter,const parametertype convergence_resolution );
            template <typename parametertype>
            void process_exactsolution(Vector<double> &solution, Solution<dim> & exact_solution, ConvergenceTable &  convergence_table_L2,  const unsigned int fe_degree, std::string & convergence_parameter,const parametertype convergence_resolution );
            template <typename parametertype>
            void process_refsolution(Vector<double> &solution,Vector<double> &solution_ref,  Vector<double> &solution_prime, Vector<double> &solution_ref_prime, ConvergenceTable & convergence_table_L2, ConvergenceTable & convergence_table_H1, ConvergenceTable & convergence_table_total,   const unsigned int fe_degree, std::string & convergence_parameter,const parametertype convergence_resolution );
            template <typename parametertype>
            void process_refsolution(Vector<double> &solution, Vector<double> &solution_ref, ConvergenceTable & convergence_table_L2,  const unsigned int fe_degree, std::string & convergence_parameter,const parametertype convergence_resolution );     
        private:
            Triangulation<dim> & triangulation_reference;
            Triangulation<dim> & triangulation_macro;
            DoFHandler<dim> &   dof_handler_macro; 
            DoFHandler<dim> &   dof_handler_reference;
            Vector<double>     solution_interpolat;
            TimerOutput &        computing_timer;


    };
}
}
#endif /* PROCESS_SOLUTION_H_ */