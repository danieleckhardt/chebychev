#include "IMEXArs3.h"

namespace MavesS{
namespace TimeIntegration
{   
    template<int dim>
    Imexars3<dim>::Imexars3(FE_Q<dim>&  fe_macro, DoFHandler<dim>&  dof_handler_macro,AffineConstraints<double>& constraints_macro, 
                                    SparseMatrix<double>& homogeneous_matrix,SparseMatrix<double>& damping_matrix, SparseMatrix<double>& mass_matrix,
                                    const double time_step_size, const double & degree_nl_v, const int example, const bool damped, const bool nonlinear)
    :
    TimeIntegrator<dim>(  fe_macro,   dof_handler_macro, constraints_macro, degree_nl_v )
    ,damped(damped)
    ,nonlinear(nonlinear)
    ,time_step_size(time_step_size)
    ,rhs_function(example)
    ,forcing_terms{}
    ,solution_macro_u_first_stage{}
    ,solution_macro_u_second_stage{}
    ,solution_macro_v_first_stage{}
    ,solution_macro_v_second_stage{}
    ,tmp{}
    ,mass_matrix(mass_matrix)
    ,homogeneous_matrix(homogeneous_matrix)
    ,damping_matrix(damping_matrix)
    ,system_matrix_macro{}
    ,solver(tol)
    {}


    template<int dim>
    void Imexars3<dim>::setup_system_macro_space()
    {
        tmp.reinit(dof_handler_macro.n_dofs()); // save vectors in the formate of U^n
        forcing_terms.reinit(dof_handler_macro.n_dofs()); // [(1 - theta) F^n-1 + theta F^n]
        pre_system_rhs_macro.reinit(dof_handler_macro.n_dofs());
        system_rhs_macro.reinit(dof_handler_macro.n_dofs());
        solution_macro_u_first_stage.reinit(dof_handler_macro.n_dofs());
        solution_macro_u_second_stage.reinit(dof_handler_macro.n_dofs());
        solution_macro_v_first_stage.reinit(dof_handler_macro.n_dofs());
        solution_macro_v_second_stage.reinit(dof_handler_macro.n_dofs());
        system_matrix_macro.reinit(homogeneous_matrix.get_sparsity_pattern());

    }
    template<int dim>
    void Imexars3<dim>::setup_system_macro_time( double new_timestep)
    {
        time_step_size = new_timestep;
    }
    template<int dim>
    void Imexars3<dim>::integrate_step(Vector<double>& solution_macro_u,Vector<double>& solution_macro_v,
                                                const Vector<double>& old_solution_macro_u, const Vector<double>& old_solution_macro_v,const double time_macro){
        // Stage 1
        assemble_system_V_first_stage(old_solution_macro_u, old_solution_macro_v, time_macro);
        this->apply_Vboundary_condition(system_matrix_macro,system_rhs_macro,solution_macro_v_first_stage);
        unsigned  n_iterations = solver.solve_time_step_size_v(system_rhs_macro, system_matrix_macro, solution_macro_v_first_stage,constraints_macro);

        solve_U_stage( solution_macro_u_first_stage,  old_solution_macro_u, solution_macro_v_first_stage, solution_macro_v_second_stage,theta*time_step_size , 0.);
        //
        //Stage 2
        assemble_system_V_second_stage(old_solution_macro_u , old_solution_macro_v,solution_macro_u_first_stage,  solution_macro_v_first_stage, time_macro);
        this->apply_Vboundary_condition(system_matrix_macro,system_rhs_macro,solution_macro_v_second_stage);
        n_iterations += solver.solve_time_step_size_v(system_rhs_macro, system_matrix_macro, solution_macro_v_second_stage,constraints_macro);

        solve_U_stage( solution_macro_u_second_stage,  old_solution_macro_u, solution_macro_v_first_stage, solution_macro_v_second_stage,(1 - 2 * theta)*time_step_size , theta*time_step_size);
        //
        // Full step
        assemble_system_V(old_solution_macro_u,solution_macro_u_first_stage,solution_macro_u_second_stage,old_solution_macro_v,solution_macro_v_first_stage, solution_macro_v_second_stage, time_macro);
        this->apply_Vboundary_condition(system_matrix_macro,system_rhs_macro,solution_macro_v);
        n_iterations += solver.solve_time_step_size_v(system_rhs_macro, system_matrix_macro, solution_macro_v,constraints_macro);
        std::cout << "    "  << n_iterations << "CG iterations per step for v." << std::endl;

        solve_U_stage( solution_macro_u,  old_solution_macro_u, solution_macro_v_first_stage, solution_macro_v_second_stage,0.5*time_step_size , 0.5*time_step_size);
        //
        //std::cout << "    "  << n_iterations << "CG iterations per step for v." << std::endl;

    }

    template <int dim>
    void Imexars3<dim>::assemble_system_V_first_stage(const Vector<double>  &old_solution_macro_u, const Vector<double>& old_solution_macro_v, const double &time_macro)
    {
        //TimerOutput::Scope t(computing_timer, "assemble_system_V");
        system_rhs_macro = 0;
        mass_matrix.vmult(system_rhs_macro, old_solution_macro_v); // M*V^n -1
        homogeneous_matrix.vmult(tmp, old_solution_macro_u);
        system_rhs_macro.add(-time_step_size*theta, tmp); //  -tau_n*theta*A*U^n-1
        rhs_function.set_time(time_macro - time_step_size);
        //rhs_function.set_time(time_macro - (1-theta)*time_step_size) ;

        VectorTools::create_right_hand_side(dof_handler_macro,
                                            QGauss<dim>(fe_macro.degree + 1),
                                            rhs_function,
                                            tmp);
        forcing_terms = tmp;
        forcing_terms *= time_step_size * theta;  // theta*tau *f(t_n+1  -  tau )

        if (nonlinear) 
        {
            this->compute_nl_term(old_solution_macro_v, tmp);
            forcing_terms.add(-time_step_size*theta ,tmp); //  theta*tau G(V_0)
        }
        system_rhs_macro += forcing_terms; 
        //
        system_matrix_macro.copy_from(mass_matrix);
        system_matrix_macro.add(theta * theta *time_step_size *time_step_size, homogeneous_matrix);  //(M + tau_n^2 theta^2 A)
        if (damped)
        {
            system_matrix_macro.add(theta *time_step_size, damping_matrix);  //(M + tau_n^2 theta^2 A + tau_n*theta*beta(t) \Delta)
        }
        constraints_macro.condense(system_matrix_macro, system_rhs_macro);
    }

    template <int dim>
    void Imexars3<dim>::assemble_system_V_second_stage(const Vector<double>  &old_solution_macro_u, const Vector<double>& old_solution_macro_v, Vector<double>& solution_macro_u_first_stage, Vector<double>& solution_macro_v_first_stage,const double &time_macro)
    {
        //TimerOutput::Scope t(computing_timer, "assemble_system_V");
        system_rhs_macro = 0;
        mass_matrix.vmult(system_rhs_macro, old_solution_macro_v); // M*v^n -1

        homogeneous_matrix.vmult(tmp, solution_macro_u_first_stage);
        system_rhs_macro.add(-time_step_size*(1 -2*theta), tmp); //  -tau*(1 - 2*theta)*A*U_1

        if (damped)
        {
            damping_matrix.vmult(tmp, solution_macro_v_first_stage);  
            system_rhs_macro.add(-time_step_size*(1 - 2*theta),tmp); // - tau *(1 - 2* theta)* beta V_1
        }

        homogeneous_matrix.vmult(tmp, old_solution_macro_u);
        system_rhs_macro.add(-time_step_size*theta, tmp); //  -tau_n*theta*A*U_0

        rhs_function.set_time(time_macro - theta*time_step_size);
        VectorTools::create_right_hand_side(dof_handler_macro,
                                            QGauss<dim>(fe_macro.degree + 1),
                                            rhs_function,
                                            tmp);
        forcing_terms = tmp;
        //forcing_terms *= time_step_size * theta;
        forcing_terms *= time_step_size * (2 - 2*theta);  // 2(1 - theta)*tau *f(t_n+1  - (1-theta) tau )

        //rhs_function.set_time(time_macro - (1- theta) *time_step_size);
        rhs_function.set_time(time_macro - time_step_size);
        VectorTools::create_right_hand_side(dof_handler_macro,
                                            QGauss<dim>(fe_macro.degree + 1),
                                            rhs_function,
                                            tmp);

        forcing_terms.add(time_step_size * (theta - 1),tmp);  // (theta -1)*tau *f(t_n+1  - tau )
        //forcing_terms.add(time_step_size*(1 - 2 * theta), tmp);

        if (nonlinear) 
        {
            this->compute_nl_term(old_solution_macro_v, tmp);
            forcing_terms.add(-time_step_size * (theta - 1) ,tmp); //  (theta - 1)*tau*G(V_0)
            this->compute_nl_term(solution_macro_v_first_stage, tmp);
            forcing_terms.add(-time_step_size * 2*(1 - theta) ,tmp); //  2*(1 - theta)*tau G(V_1)
        }


        system_rhs_macro += forcing_terms; // 
        //
        system_matrix_macro.copy_from(mass_matrix);
        system_matrix_macro.add(theta * theta *time_step_size *time_step_size, homogeneous_matrix);  //(M + tau_n^2 theta^2 A)
        if (damped)
        {
            system_matrix_macro.add(theta *time_step_size, damping_matrix);  //(M + tau_n^2 theta^2 A + tau_n*theta*beta(t) \Delta)
        }
        constraints_macro.condense(system_matrix_macro, system_rhs_macro);
    }

    template <int dim>
    void Imexars3<dim>::assemble_system_V(const Vector<double>  &old_solution_macro_u,Vector<double>  &solution_macro_u_first_stage,Vector<double>  &solution_macro_u_second_stage, const Vector<double>  &old_solution_macro_v, Vector<double>& solution_macro_v_first_stage,Vector<double>& solution_macro_v_second_stage, const double &time_macro)
    {
        //TimerOutput::Scope t(computing_timer, "assemble_system_V");
        system_rhs_macro = 0;
        mass_matrix.vmult(system_rhs_macro, old_solution_macro_v); // M*V^n -1
        homogeneous_matrix.vmult(tmp, solution_macro_u_first_stage);
        system_rhs_macro.add(-time_step_size*0.5, tmp); //  -tau_n*0.5*A*U_1
        homogeneous_matrix.vmult(tmp, solution_macro_u_second_stage);
        system_rhs_macro.add(-time_step_size*0.5, tmp); //  -tau_n*0.5*A*U_2

        if (damped)
        {
            damping_matrix.vmult(tmp, solution_macro_v_first_stage);  
            system_rhs_macro.add(-time_step_size*0.5,tmp);
            damping_matrix.vmult(tmp, solution_macro_v_second_stage);  
            system_rhs_macro.add(-time_step_size*0.5,tmp); // -tau *0.5*( BV_1 + BV_2)
        }

        rhs_function.set_time(time_macro - theta*time_step_size);
        VectorTools::create_right_hand_side(dof_handler_macro,
                                            QGauss<dim>(fe_macro.degree + 1),
                                            rhs_function,
                                            tmp);
        forcing_terms = tmp; // f(t_n+1 - theta * tau)
        forcing_terms *= time_step_size*0.5;
        rhs_function.set_time(time_macro - (1-theta)*time_step_size);
        VectorTools::create_right_hand_side(dof_handler_macro,
                                            QGauss<dim>(fe_macro.degree + 1),
                                            rhs_function,
                                            tmp);
        forcing_terms.add(time_step_size*0.5,tmp);// f(t_n+1 - (1-theta) * tau)

        if (nonlinear) 
        {
            this->compute_nl_term(solution_macro_v_first_stage, tmp);
            forcing_terms.add(-time_step_size*0.5 ,tmp); 
            this->compute_nl_term(solution_macro_v_second_stage, tmp);
            forcing_terms.add(-time_step_size*0.5 ,tmp); 

        } 
        system_rhs_macro += forcing_terms; // tau * 0.5 * (f(t_n+1  - (1-theta) tau ) + f(t_n+1  - theta*tau ) + G(V_1) + G(V_2) )
        //
        system_matrix_macro.copy_from(mass_matrix);
        constraints_macro.condense(system_matrix_macro, system_rhs_macro);
    }
    template <int dim>
    void Imexars3<dim>::solve_U_stage( Vector<double>& solution_macro_u, const Vector<double>& old_solution_macro_u,Vector<double>& solution_macro_v_first_stage, Vector<double>& solution_macro_v_second_stage,  const double time_macro_step_first_factor,const double time_macro_step_second_factor)
    {
        solution_macro_u = old_solution_macro_u;
        solution_macro_u.add(time_macro_step_first_factor,solution_macro_v_first_stage );
        solution_macro_u.add( time_macro_step_second_factor, solution_macro_v_second_stage);
    }

    template class Imexars3<1>;
    template class Imexars3<2>;
}
}

