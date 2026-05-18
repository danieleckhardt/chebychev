#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>

#include<deal.II/meshworker/mesh_loop.h>

#include <cmath>
#include "../../data/Coefficients.h"
#include "../../generating/MicroProblem.h"

namespace pt = boost::property_tree;

using namespace dealii;

int main()//int argc, char *argv[]) // ändern?
{
    const unsigned int dim = 2;
    using namespace dealii;
    using namespace MavesS;
    using namespace Data;
    using namespace Microproblem;
    {
    double epsilon = std::pow(2,-25);
    double delta   = std::pow(2,-23);


    unsigned int level_micro = 3;
    Triangulation<dim> triangulation_macro;
    GridGenerator::hyper_cube(triangulation_macro, 0, 1);
    triangulation_macro.refine_global(2);
    DoFHandler<dim>    dof_handler_macro(triangulation_macro);
    FE_Q<dim>          fe_macro(1);
    AffineConstraints<double> constraints_macro;
    SparseMatrix<double> homogeneous_matrix;
    SparsityPattern sparsity_pattern_macro;

    dof_handler_macro.distribute_dofs(fe_macro);
    constraints_macro.clear();
    DoFTools::make_hanging_node_constraints(dof_handler_macro, constraints_macro);
    constraints_macro.close();
    DynamicSparsityPattern dsp(dof_handler_macro.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_macro, dsp, constraints_macro, true);
    sparsity_pattern_macro.copy_from(dsp);
    homogeneous_matrix.reinit(sparsity_pattern_macro);
    TimerOutput        computing_timer(std::cout,
                      TimerOutput::never,
                      TimerOutput::wall_times);



    GeneratingHomCoeff<dim>   genHomCoeff(epsilon,delta,false, -2, level_micro, 2,dof_handler_macro, fe_macro, constraints_macro, computing_timer);
    genHomCoeff.setup_system_and_triang_micro();
    genHomCoeff.assemble_B_H(homogeneous_matrix);
    double test_value = homogeneous_matrix.frobenius_norm();
    std::cout<< test_value << std::endl; // Test against approximation of the exact homogeneous coefficient
    }
    return 0;
}