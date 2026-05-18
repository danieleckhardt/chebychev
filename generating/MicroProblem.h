#ifndef MICROPROBLEM_H_ 
#define MICROPROBLEM_H_

#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include<deal.II/meshworker/mesh_loop.h>

#include <cmath>
#include "../data/Coefficients.h"

using namespace dealii;

namespace MavesS // Multi-Waves-Scale
{
using namespace Data;
namespace Microproblem
{
    template <int dim>
    class GeneratingHomCoeff
    {
        public:
            Triangulation<dim> triangulation_micro;
            GeneratingHomCoeff(double epsilon,double delta, bool periodic,int example_config, unsigned int level_micro, unsigned int order_micro, DoFHandler<dim>& dof_handler_macro, 
                                FE_Q<dim> &fe_macro,  AffineConstraints<double>&  constraints_macro,  TimerOutput &  computing_timer);
            //Declare the Matricies and Vectors for the Micro system + Triangulation 
            void setup_system_and_triang_micro();

            // Assemble the homogeneous matrics
            void assemble_B_H(SparseMatrix<double> &homogeneous_matrix);
            void assemble_damping_with_cellproblem(SparseMatrix<double> &damp_matrix);
            void assemble_longtime(SparseMatrix<double> &homogeneous_matrix);
            void assemble_B_H_symmetric(SparseMatrix<double> &homogeneous_matrix);
            void assemble_B_H_origin(SparseMatrix<double> &homogeneous_matrix);
            void assemble_B_H_homogenized(SparseMatrix<double> &homogeneous_matrix);
            void assemble_damping_grad(SparseMatrix<double> &damping_matrix);
            TimerOutput  & computing_timer;

        private:
            double                                 epsilon;
            double                                 delta;
            bool                                   periodic;
            int                                    example_config;
            unsigned int                           level_micro;
            FE_Q<dim>                              fe_micro;
            DoFHandler<dim>                        dof_handler_micro;
            FE_Q<dim>&                             fe_macro;
            AffineConstraints<double>&             constraints_macro;
            AffineConstraints<double>              constraints_micro;
            DoFHandler<dim>&                       dof_handler_macro;
            const MappingQ1<dim>                   mapping;
            SparsityPattern                        sparsity_pattern_micro;
            const QGauss<dim>                      quadrature;
            Coefficient_a<dim>                     coefficient;
            Coefficient_a_matrix_homogenized<dim>  coefficient_matrix_homogenized;
            Coefficient_beta<dim>                  coefficient_beta;

            // Setup the Mirco-System
            void assemble_solve_micro(Point<dim> p, Tensor<1,dim> gradphi, Vector<double>& solution );
            void solve_micro(Vector<double>& rhs, SparseMatrix<double>& matrix, Vector<double>& loesung);

    };

  template <int dim>
  struct ScratchDataMacro
  {
    ScratchDataMacro(const Mapping<dim>&  mapping,
                const FiniteElement<dim>& fe_macro,
                const QGauss<dim>&        quadrature,
                const UpdateFlags         update_flags = 
                                                 update_values |
                                                 update_gradients |
                                                 update_quadrature_points |
                                                 update_JxW_values
    )
      : fe_values_macro(mapping, fe_macro, quadrature, update_flags)

    {}
 
 
    ScratchDataMacro(const ScratchDataMacro<dim> &scratch_data_macro)
      : fe_values_macro(scratch_data_macro.fe_values_macro.get_mapping(),
                  scratch_data_macro.fe_values_macro.get_fe(),
                  scratch_data_macro.fe_values_macro.get_quadrature(),
                  scratch_data_macro.fe_values_macro.get_update_flags())

    {}
 
    FEValues<dim>          fe_values_macro;
  };

  struct CopyDataMacro
  {
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> local_dof_indices; 
    template <class Iterator>
    void reinit(const Iterator &cell, unsigned int dofs_per_cell)
    {
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
    }
  };

}
}
 #endif /* MICROPROBLEM_H_ */