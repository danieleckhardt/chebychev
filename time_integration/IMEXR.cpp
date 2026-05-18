#include "IMEXR.h"

namespace MavesS{
namespace TimeIntegration
{   
    template<int dim>
    ImexR<dim>::ImexR(FE_Q<dim>&  fe_macro, DoFHandler<dim>&  dof_handler_macro,AffineConstraints<double>& constraints_macro, 
                                    SparseMatrix<double>& homogeneous_matrix, SparseMatrix<double>& damping_matrix,SparseMatrix<double>& mass_matrix,
                                    const double time_step_size,const  double & degree_nl_v, const int example, const bool damped, const bool nonlinear)
    :
    TimeIntegrator<dim>(  fe_macro,   dof_handler_macro, constraints_macro, degree_nl_v)
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
    void ImexR<dim>::setup_system_macro_space()
    {
        tmp.reinit(dof_handler_macro.n_dofs()); // save vectors in the formate of U^n
        forcing_terms.reinit(dof_handler_macro.n_dofs()); // [(1 - theta) F^n-1 + theta F^n]
        pre_system_rhs_macro.reinit(dof_handler_macro.n_dofs());
        system_rhs_macro.reinit(dof_handler_macro.n_dofs());
        solution_macro_v_half.reinit(dof_handler_macro.n_dofs());
        system_matrix_macro.reinit(homogeneous_matrix.get_sparsity_pattern());
    }
    template<int dim>
    void ImexR<dim>::setup_system_macro_time( double new_timestep)
    {
        time_step_size = new_timestep;
    }
    template<int dim>
    void ImexR<dim>::integrate_step(Vector<double>& solution_macro_u,Vector<double>& solution_macro_v,const Vector<double>& old_solution_macro_u, const Vector<double>& old_solution_macro_v,const double time_macro)
    {
        assemble_system_V_half(old_solution_macro_u, old_solution_macro_v, time_macro);
        this->apply_Vboundary_condition(system_matrix_macro,system_rhs_macro,solution_macro_v_half);
        n_iterations = solver.solve_time_step_size_v(system_rhs_macro, system_matrix_macro, solution_macro_v_half,constraints_macro);
        mv_count = n_iterations;
        assemble_system_V(old_solution_macro_u,old_solution_macro_v, solution_macro_v_half, time_macro);
        this->apply_Vboundary_condition(system_matrix_macro,system_rhs_macro,solution_macro_v);
        int n_iterations_temp = solver.solve_time_step_size_v(system_rhs_macro, system_matrix_macro, solution_macro_v,constraints_macro);
        mv_count += n_iterations_temp;
        system_rhs_macro = 0;
        system_rhs_macro.add(time_step_size , solution_macro_v_half);
        system_rhs_macro +=  old_solution_macro_u; // U^n-1
        solution_macro_u = system_rhs_macro;
    }

    template <int dim>
    void ImexR<dim>::assemble_system_V_half( const Vector<double>  &old_solution_macro_u, const Vector<double>& old_solution_macro_v, const double &time_macro)
    {
        //TimerOutput::Scope t(computing_timer, "assemble_system_V");
        system_rhs_macro = 0;
        mass_matrix.vmult(system_rhs_macro, old_solution_macro_v); // M*V^n -1
        mv_count += 1;
        homogeneous_matrix.vmult(tmp, old_solution_macro_u);
        mv_count += 1;
        system_rhs_macro.add(-time_step_size*theta, tmp); //  -tau_n*theta*A*U^n-1
        //assemble_rhs(rhs_function,tmp);
        rhs_function.set_time(time_macro);
        VectorTools::create_right_hand_side(dof_handler_macro,
                                            QGauss<dim>(fe_macro.degree + 1),
                                            rhs_function,
                                            tmp);
        forcing_terms = tmp; // F^n
        forcing_terms *= time_step_size * theta*theta; 
        rhs_function.set_time(time_macro - time_step_size);
        VectorTools::create_right_hand_side(dof_handler_macro,
                                        QGauss<dim>(fe_macro.degree + 1),
                                        rhs_function,
                                        tmp);
        forcing_terms.add(time_step_size * (1. - theta)*(1. - theta), tmp);
        if (nonlinear) 
        {
            this->compute_nl_term(old_solution_macro_v, tmp);
            forcing_terms.add( -time_step_size*theta ,tmp);

        }
        system_rhs_macro += forcing_terms; //  +[tau_n *theta *theta (F^n ) + tau_n *(1-theta) * (1-theta)(F^n-1) - tau_n *theta*nonlinear^n-1(v^n-1))]
        system_matrix_macro.copy_from(mass_matrix);
        system_matrix_macro.add(theta * theta *time_step_size *time_step_size, homogeneous_matrix);  //(M + tau_n^2 theta^2 A)
        if (damped)
        {
            system_matrix_macro.add(theta *time_step_size , damping_matrix);  //(M + tau_n^2 theta^2 A + tau_n*theta*beta(t) \Delta)
        }
        constraints_macro.condense(system_matrix_macro, system_rhs_macro);

    }

    template <int dim>
    void ImexR<dim>::assemble_system_V( const Vector<double>  &old_solution_macro_u, const Vector<double>  &old_solution_macro_v, Vector<double>& solution_macro_v_half, const double &time_macro)
    {
        //TimerOutput::Scope t(computing_timer, "assemble_system_V");
        system_rhs_macro = 0;
        mass_matrix.vmult(system_rhs_macro, solution_macro_v_half); 
        mv_count += 1;
        system_rhs_macro *= 2; // 2*M*V^n-1/2
        mass_matrix.vmult(tmp, old_solution_macro_v);
        mv_count += 1;
        system_rhs_macro.add(-1, tmp); // 2*M*V^n-1/2 - M*vn-1
        if (nonlinear) 
        {
            this->compute_nl_term(solution_macro_v_half, tmp);
            system_rhs_macro.add( -time_step_size ,tmp);
            this->compute_nl_term(old_solution_macro_v, tmp);
            system_rhs_macro.add( time_step_size ,tmp);
        }
        system_matrix_macro.copy_from(mass_matrix);
        constraints_macro.condense(system_matrix_macro, system_rhs_macro);

    }

    template class ImexR<1>;
    template class ImexR<2>;
}
}

