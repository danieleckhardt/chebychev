#include "solving.h"
#include <cmath>

namespace MavesS // Multi-Waves-Scale
{
namespace Solving
{   
    using namespace dealii;
    template <int dim>
    Solver<dim>::Solver(double& tol)
    :
        tol(tol)
    {}

    template <int dim>
    unsigned int Solver<dim>::solve_time_step_size_v(Vector<double> &system_rhs_macro, SparseMatrix<double>&system_matrix_macro, 
                                                            Vector<double> &solution,AffineConstraints<double>& constraints_macro){
        SolverControl            solver_control(1000, tol * system_rhs_macro.l2_norm());
        SolverCG<Vector<double>> cg(solver_control);

        PreconditionSSOR<SparseMatrix<double>> preconditioner;
        preconditioner.initialize(system_matrix_macro, 1.2); 
        cg.solve(system_matrix_macro, solution, system_rhs_macro, preconditioner);
        constraints_macro.distribute(solution); // settig the right values
        return solver_control.last_step();
    }

    template <int dim>
    void Solver<dim>::solve_time_step_u(Vector<double>& system_rhs_macro, SparseMatrix<double> &system_matrix_macro,
                                                    Vector<double>& solution,AffineConstraints<double>& constraints_macro){
        SolverControl            solver_control(1000, tol * system_rhs_macro.l2_norm());
        SolverCG<Vector<double>> cg(solver_control);

        PreconditionSSOR<SparseMatrix<double>> preconditioner;
        preconditioner.initialize(system_matrix_macro, 1.2);

        cg.solve(system_matrix_macro, solution, system_rhs_macro, preconditioner);

        constraints_macro.distribute(solution); // settig the right values

    }


template class Solver<1>;
template class Solver<2>;

} 
}