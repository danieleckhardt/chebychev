#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/matrix_tools.h>

#include "../../data/ExactSolution.h"
#include "../../data/InitialConditions.h"

#include "../../time_integration/imp_Midpoint.h"

#include <iostream>

using namespace dealii;
using namespace MavesS;
using namespace Data;
using namespace TimeIntegration;

template <int dim>
class testspace1
{
    public:
        testspace1();
        double run();
    private:
        void setup_system();
        void cycle_time_steps();
        void process_solution(Solution<dim>& exact_solution);
        double EoC();

        Triangulation<dim> triangulation;

        DoFHandler<dim>    dof_handler;
        FE_Q<dim>          fe; 
        SparsityPattern    sparsity_pattern;

        SparseMatrix<double> system_matrix;
        SparseMatrix<double> laplace_matrix;
        SparseMatrix<double> homogeneous_matrix;
        SparseMatrix<double> damping_matrix;
        SparseMatrix<double> mass_matrix;
        
        AffineConstraints<double> constraints;

        Vector<double> solution_u;
        Vector<double> old_solution_u;
        Vector<double> solution_v;
        Vector<double> old_solution_v;

        double time_step_size;
        double time;
        double eoc = 0;

        impMidpoint <dim> iM_scheme;

        std::list<double>  error_list;
        std::list<double>  time_step_size_list;

};

template<int dim>
testspace1<dim>::testspace1()
:   
    dof_handler(triangulation)
   ,fe(1)
   ,iM_scheme(fe, dof_handler,constraints,homogeneous_matrix,damping_matrix,mass_matrix,time_step_size,0,-1,false,false)
{}

template<int dim>
void testspace1<dim>::setup_system()
{

    dof_handler.distribute_dofs(fe);

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
    sparsity_pattern.copy_from(dsp);

    laplace_matrix.reinit(sparsity_pattern);
    homogeneous_matrix.reinit(sparsity_pattern);
    damping_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);

    solution_u.reinit(dof_handler.n_dofs());
    old_solution_u.reinit(dof_handler.n_dofs());
    solution_v.reinit(dof_handler.n_dofs());
    old_solution_v.reinit(dof_handler.n_dofs());

    MatrixCreator::create_laplace_matrix(dof_handler, QGauss<dim>(fe.degree + 1), laplace_matrix);
    MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(fe.degree + 1), mass_matrix);
    homogeneous_matrix.copy_from(laplace_matrix);
    damping_matrix.copy_from(damping_matrix);
    homogeneous_matrix *= 0.5;
    damping_matrix *= 0.;
}

template<int dim>
void testspace1<dim>::cycle_time_steps()
{
    time = 0.;
    while(time < 1.0)       
    {
        time += time_step_size;
        std::cout.setstate(std::ios_base::failbit);
        iM_scheme.integrate_step(solution_u,solution_v,old_solution_u, old_solution_v, time);
        std::cout.clear();
        old_solution_u = solution_u;
        old_solution_v = solution_v;
    }

}
template<int dim>   
void testspace1<dim>::process_solution(Solution<dim>& exact_solution)
{
    Vector<float> difference_per_cell(triangulation.n_active_cells());

    VectorTools::integrate_difference(dof_handler,
                                  solution_u,
                                  exact_solution,
                                  difference_per_cell,
                                  QGauss<dim>(fe.degree + 1), 
                                  VectorTools::H1_norm);
    const double H1_error =
        VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::H1_norm);
    error_list.push_back(H1_error);
}

template<int dim>
double testspace1<dim>::EoC()
{
    int  average_number = error_list.size() - 2;
    double tmp_error;
    double tmp_time_step_size;
    while(error_list.size()>2)
    {
        tmp_error = error_list.front();
        tmp_time_step_size  = time_step_size_list.front();
        error_list.pop_front();
        time_step_size_list.pop_front();
        eoc += std::log(tmp_error/error_list.front()) / std::log(tmp_time_step_size/time_step_size_list.front()); // siehe Wikipedia EOC
    }

    return eoc/average_number;
}

template<int dim>
double testspace1<dim>::run()
{
    const unsigned int initial_refinement    = 1;
    const double       refinement_steps      = 8.;
    unsigned int count_refinement_step       = 0;
    time_step_size                           = 1/100.;
    double grid_width                        = 1/2.;

    Solution<dim>      exact_solution(-1);
    exact_solution.set_time(1.0);

    GridGenerator::hyper_cube(triangulation,0,1);
    triangulation.refine_global(initial_refinement);
    iM_scheme.setup_system_macro_time(time_step_size);

    while(count_refinement_step < refinement_steps+1)
    {
        setup_system();

        iM_scheme.setup_system_macro_space();

        VectorTools::interpolate(dof_handler,
                                InitialCondition<dim>(InitialCondition<dim>::Components::U, -1),
                                old_solution_u);
        VectorTools::interpolate(dof_handler,
                                InitialCondition<dim>(InitialCondition<dim>::Components::V, -1),
                                old_solution_v);
        solution_u = old_solution_u;
        solution_v = old_solution_v;

        cycle_time_steps();
        process_solution(exact_solution);
        time_step_size_list.push_back(grid_width);

        ++count_refinement_step;
        triangulation.refine_global(1);
        grid_width *= 1/2.;
    }
    return eoc = this->EoC();

}

int main()
{
const unsigned int dim = 1;
using namespace dealii;
using namespace MavesS;
using namespace Data;
using namespace Solving;

double eoc; 
double order = 1.;

testspace1<dim> test;
eoc = test.run();

if(std::abs(order - eoc) < 0.1 )
    std::cout<< 1;
else
    std::cout<< 0 << "  EOC = "<< eoc ;
}

