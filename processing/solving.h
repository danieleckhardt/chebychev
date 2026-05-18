#ifndef SOLVING_H_
#define SOLVING_H_


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
 
#include <deal.II/lac/generic_linear_algebra.h>

/* namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
 using namespace dealii::LinearAlgebraPETSc;
} // namespace LA */

#include <deal.II/base/utilities.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
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
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>


#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

namespace MavesS // Multi-Waves-Scale
{
namespace Solving{
using namespace dealii;

template <int dim>
class Solver
  {
    public:
        Solver(double& tol);
        unsigned int solve_time_step_size_v(Vector<double> &system_rhs_macro, SparseMatrix<double>&system_matrix_macro, 
                                                            Vector<double> &solution,AffineConstraints<double>& constraints_macro);

        void solve_time_step_u(Vector<double>& system_rhs_macro, SparseMatrix<double> &system_matrix_macro,
                                                    Vector<double>& solution,AffineConstraints<double>& constraints_macro);   
    private:
    double& tol;
  };

}
}
#endif /* SOLVING_H_ */