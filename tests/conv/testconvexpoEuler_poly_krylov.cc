#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/utilities.h>

#include "../../data/ExactSolution.h"
#include "../../data/InitialConditions.h"

#include "../../time_integration/expoEuler.h"

#include <iostream>

using namespace dealii;
using namespace MavesS;
using namespace Data;
using namespace TimeIntegration;

template <int dim>
class testexpoEulerkrylov
{
    public:
        testexpoEulerkrylov();
        double run();
    private:
        void setup_system();
        void cycle_time_steps(const double &time_step_size);
        void process_solution(Solution<dim>& exact_solution);
        double EoC();

        Triangulation<dim> triangulation;

        DoFHandler<dim>    dof_handler;
        FE_Q<dim>          fe; 
        SparsityPattern    sparsity_pattern;

        SparseMatrix<double> system_matrix;
        SparseMatrix<double> laplace_matrix;
        SparseMatrix<double> damping_matrix;
        SparseMatrix<double> homogeneous_matrix;
        SparseMatrix<double> mass_matrix;
        
        AffineConstraints<double> constraints;

        Vector<double> solution_u;
        Vector<double> old_solution_u;
        Vector<double> solution_v;
        Vector<double> old_solution_v;

        double time_step_size;
        double time;
        double eoc = 0.;

        ExpoEuler<dim> expoeuler_scheme;

        std::list<double>  error_list;
        std::list<double>  time_step_size_list;

};

template<int dim>
testexpoEulerkrylov<dim>::testexpoEulerkrylov()
:   
    dof_handler(triangulation)
   ,fe(1)
   ,expoeuler_scheme(fe, dof_handler,constraints,homogeneous_matrix,damping_matrix,mass_matrix,"krylov", time_step_size,0,-1,true,false)
{}

template<int dim>
void testexpoEulerkrylov<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(
        dof_handler,               
        0,                         
        Functions::ZeroFunction<dim>(),        
        constraints);       
    VectorTools::interpolate_boundary_values(
        dof_handler,               
        1,                         
        Functions::ZeroFunction<dim>(),        
        constraints);     
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
    sparsity_pattern.copy_from(dsp);

    damping_matrix.reinit(sparsity_pattern);
    homogeneous_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);

    solution_u.reinit(dof_handler.n_dofs());
    old_solution_u.reinit(dof_handler.n_dofs());
    solution_v.reinit(dof_handler.n_dofs());
    old_solution_v.reinit(dof_handler.n_dofs());

    MatrixCreator::create_laplace_matrix(dof_handler, QGauss<dim>(fe.degree + 1), laplace_matrix);
    MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(fe.degree + 1), mass_matrix);
    homogeneous_matrix.copy_from(laplace_matrix);
    homogeneous_matrix *= 0.5;
    damping_matrix.copy_from(laplace_matrix);
    damping_matrix *= 0.; // beta
}

template<int dim>
void testexpoEulerkrylov<dim>::cycle_time_steps(const double &time_step_size)
{
    time = 0.;
    expoeuler_scheme.setup_system_macro_time(time_step_size);
    while(time < 1.0)       
    {
        time += time_step_size;
        std::cout.setstate(std::ios_base::failbit);
        expoeuler_scheme.integrate_step(solution_u,solution_v,old_solution_u, old_solution_v, time);
        std::cout.clear();
        old_solution_u = solution_u;
        old_solution_v = solution_v;
    }

}
template<int dim>   
void testexpoEulerkrylov<dim>::process_solution(Solution<dim>& exact_solution)
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
double testexpoEulerkrylov<dim>::EoC()
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
double testexpoEulerkrylov<dim>::run()
{
    const unsigned int initial_refinement    = 6;
    const double       refinement_steps      = 6.;
    unsigned int count_refinement_step       = 0;
    time_step_size                           = 1./2;

    Solution<dim>      exact_solution(-1);
    exact_solution.set_time(1.0);


    GridGenerator::hyper_cube(triangulation,0,1);
    triangulation.refine_global(initial_refinement);
    setup_system();
    
    expoeuler_scheme.set_max_iteration_krylov(dof_handler.n_dofs()*2);
    expoeuler_scheme.setup_system_macro_space();

    while(count_refinement_step < refinement_steps+1)
    {
        VectorTools::interpolate(dof_handler,
                                InitialCondition<dim>(InitialCondition<dim>::Components::U, -1),
                                old_solution_u);
        VectorTools::interpolate(dof_handler,
                                InitialCondition<dim>(InitialCondition<dim>::Components::V, -1),
                                old_solution_v);
        solution_u = old_solution_u;
        solution_v = old_solution_v;
        cycle_time_steps(time_step_size);
        process_solution(exact_solution);
        time_step_size_list.push_back(time_step_size);

        ++count_refinement_step;  
        time_step_size   *= 1/2.; 
    }
    return eoc = this->EoC();

}

int main(int argc, char **argv)
{
const unsigned int dim = 1;
using namespace dealii;
using namespace MavesS;
using namespace Data;
using namespace Solving;
Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);


double eoc; 
double order = 1.;

testexpoEulerkrylov<dim> test;
eoc = test.run();
if(std::abs(order - eoc) < 0.1 )
    std::cout<< 1;
else
    std::cout<< 0 << "  EOC = "<< eoc ;
}

