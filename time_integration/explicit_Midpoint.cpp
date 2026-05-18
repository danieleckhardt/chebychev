#include "explicit_Midpoint.h"

namespace MavesS{
namespace TimeIntegration
{   
    template<int dim>
    expMidpoint<dim>::expMidpoint(FE_Q<dim>&  fe_macro, DoFHandler<dim>&  dof_handler_macro,AffineConstraints<double>& constraints_macro, 
                                    SparseMatrix<double>& homogeneous_matrix,SparseMatrix<double>& damping_matrix, SparseMatrix<double>& mass_matrix,
                                    const double time_step_size, const double & degree_nl_v, const int example, const bool damped, const bool nonlinear)
    :
    TimeIntegrator<dim>(  fe_macro,   dof_handler_macro, constraints_macro,degree_nl_v)
    ,damped(damped)
    ,nonlinear(nonlinear)
    ,time_step_size(time_step_size)
    ,rhs_function(example)
    ,forcing_terms{}
    ,solution_macro_v_half{}
    ,tmp{}
    ,mass_matrix(mass_matrix)
    ,homogeneous_matrix(homogeneous_matrix)
    ,damping_matrix(damping_matrix)
    ,system_matrix_macro{}
    ,solver(tol)
    {}


    template<int dim>
    void expMidpoint<dim>::setup_system_macro_space()
    {
        tmp.reinit(dof_handler_macro.n_dofs()); // save vectors in the formate of U^n
        forcing_terms.reinit(dof_handler_macro.n_dofs()); // [(1 - theta) F^n-1 + theta F^n]
        system_rhs_macro.reinit(dof_handler_macro.n_dofs());
        solution_macro_v_half.reinit(dof_handler_macro.n_dofs());
        system_matrix_macro.reinit(homogeneous_matrix.get_sparsity_pattern());
    }
    template<int dim>
    void expMidpoint<dim>::setup_system_macro_time( double new_timestep)
    {
        time_step_size = new_timestep;
    }
    template<int dim>
    void expMidpoint<dim>::integrate_step(Vector<double>& solution_macro_u,Vector<double>& solution_macro_v, const Vector<double>& old_solution_macro_u, const Vector<double>& old_solution_macro_v,const double time_macro){

        assemble_system_V_half(old_solution_macro_u, old_solution_macro_v, time_macro);
        this->apply_Vboundary_condition(system_matrix_macro,system_rhs_macro,solution_macro_v_half);
        solver.solve_time_step_size_v(system_rhs_macro, system_matrix_macro, solution_macro_v_half,constraints_macro);

        assemble_system_V(old_solution_macro_u,old_solution_macro_v, solution_macro_v_half, time_macro);
        this->apply_Vboundary_condition(system_matrix_macro,system_rhs_macro,solution_macro_v);
        unsigned  n_iterations = solver.solve_time_step_size_v(system_rhs_macro, system_matrix_macro, solution_macro_v,constraints_macro);

        std::cout << "    "  << n_iterations << "CG iterations per step for v." << std::endl;

        system_rhs_macro = 0;
        system_rhs_macro.add(time_step_size , solution_macro_v_half); // tau   V^n+1/2
        system_rhs_macro +=  old_solution_macro_u; // U^n-1
        solution_macro_u = system_rhs_macro;


    }
   
    template <int dim>
    void expMidpoint<dim>::assemble_system_V_half( const Vector<double>  &old_solution_macro_u, const Vector<double>& old_solution_macro_v, const double &time_macro)
    {
  
        system_rhs_macro = 0;
        mass_matrix.vmult(system_rhs_macro, old_solution_macro_v); // M*V^n -1
        homogeneous_matrix.vmult(tmp, old_solution_macro_u);
        system_rhs_macro.add(-time_step_size*theta, tmp); //  -tau_n*theta*A*U^n-1
        if (damped)
        {
            damping_matrix.vmult(tmp, old_solution_macro_v);
            system_rhs_macro.add(-time_step_size*theta, tmp); //  -tau_n*theta*B*v^n-1
        }
        rhs_function.set_time(time_macro - time_step_size);
        VectorTools::create_right_hand_side(dof_handler_macro,
                                        QGauss<dim>(fe_macro.degree + 1),
                                        rhs_function,
                                        tmp);
        forcing_terms = tmp; // F^n
        forcing_terms *= time_step_size * theta; 
        if (nonlinear) 
        {
            this->compute_nl_term(old_solution_macro_v, tmp);
            forcing_terms.add( -time_step_size*theta ,tmp);

        }
        system_rhs_macro += forcing_terms; //  +[tau_n *theta *theta (F^n ) + tau_n *(1-theta) * (1-theta)(F^n-1) - tau_n *theta*nonlinear^n-1(v^n-1))]
        system_matrix_macro.copy_from(mass_matrix);
        constraints_macro.condense(system_matrix_macro, system_rhs_macro);

    }

    template <int dim>
    void expMidpoint<dim>::assemble_system_V(  const Vector<double>  &old_solution_macro_u, const Vector<double>  &old_solution_macro_v, Vector<double>& solution_macro_v_half, const double &time_macro)
    {
        //TimerOutput::Scope t(computing_timer, "assemble_system_V");
        system_rhs_macro = 0.;
        mass_matrix.vmult(system_rhs_macro, old_solution_macro_v); // M*V^n -1
        homogeneous_matrix.vmult(tmp, old_solution_macro_u);
        system_rhs_macro.add(-time_step_size, tmp); //  -tau_n*A*U^n-1
        homogeneous_matrix.vmult(tmp, old_solution_macro_v);
        system_rhs_macro.add(-time_step_size*time_step_size*theta, tmp); //  -tau_n^2*theta*A*v^n-1/2

        if (damped)
        {
            damping_matrix.vmult(tmp, solution_macro_v_half);
            system_rhs_macro.add(-time_step_size, tmp);  // tau_n*theta*Bv ^n-1/2
        }
        rhs_function.set_time(time_macro - time_step_size/2.);
        VectorTools::create_right_hand_side(dof_handler_macro,
                                        QGauss<dim>(fe_macro.degree + 1),
                                        rhs_function,
                                        tmp);
        forcing_terms = tmp; // F^n
        forcing_terms *= time_step_size ; 
        if (nonlinear) 
        {
            this->compute_nl_term(solution_macro_v_half, tmp);
            forcing_terms.add( -time_step_size ,tmp);
        }
        system_rhs_macro += forcing_terms; //  +[ tau_n * (F^n-1/2) - tau_n *nonlinear^n-1(v^n-1))]


        system_matrix_macro.copy_from(mass_matrix);

        constraints_macro.condense(system_matrix_macro, system_rhs_macro);

    }

    template class expMidpoint<1>;
    template class expMidpoint<2>;
}
}

