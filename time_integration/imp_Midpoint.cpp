#include "imp_Midpoint.h"

namespace MavesS{
namespace TimeIntegration
{   
    template<int dim>
    impMidpoint<dim>::impMidpoint(FE_Q<dim>&  fe_macro, DoFHandler<dim>&  dof_handler_macro,AffineConstraints<double>& constraints_macro, 
                                            SparseMatrix<double>& homogeneous_matrix, SparseMatrix<double>& damping_matrix,
                                            SparseMatrix<double>& mass_matrix,const double time_step_size,const double & degree_nl_v, const int example, 
                                            const bool damped, const bool nonlinear)
    :
    TimeIntegrator<dim>(fe_macro,dof_handler_macro,constraints_macro,degree_nl_v)
    ,damped(damped)
    ,nonlinear(nonlinear)
    ,time_step_size(time_step_size)
    ,rhs_function(example)
    ,forcing_terms{}
    ,solution_macro_v_update{}
    ,tmp{}
    ,mass_matrix(mass_matrix)
    ,homogeneous_matrix(homogeneous_matrix)
    ,damping_matrix(damping_matrix)
    ,tmp_matrix{} 
    ,system_matrix_macro{}
    ,solver(tol)
    {}

    template<int dim>
    void impMidpoint<dim>::setup_system_macro_space()
    {
        tmp.reinit(dof_handler_macro.n_dofs()); // save vectors in the formate of U^n
        forcing_terms.reinit(dof_handler_macro.n_dofs()); // [(1 - theta) F^n-1 + theta F^n]
        pre_system_rhs_macro.reinit(dof_handler_macro.n_dofs());
        system_rhs_macro.reinit(dof_handler_macro.n_dofs());
        solution_macro_v_update.reinit(dof_handler_macro.n_dofs());
        tmp_matrix.reinit(homogeneous_matrix.get_sparsity_pattern());
        system_matrix_macro.reinit(homogeneous_matrix.get_sparsity_pattern());
        old_step_matrix.reinit(homogeneous_matrix.get_sparsity_pattern());
        update_matrix.reinit(homogeneous_matrix.get_sparsity_pattern());
    }
    template<int dim>
    void impMidpoint<dim>::setup_system_macro_time( double new_timestep)
    {
        time_step_size = new_timestep;
    }
    
    template<int dim>
    void impMidpoint<dim>::integrate_step(Vector<double>& solution_macro_u,Vector<double>& solution_macro_v, const Vector<double>& old_solution_macro_u, const Vector<double>& old_solution_macro_v,const double time_macro)
    {
        initial_rhs_norm = 0.;
        int counter = 0;
        first_iteration  = true;
        pre_assemble_system_V( old_solution_macro_u, old_solution_macro_v, time_macro);
        do
        {
            assemble_system_V_new(solution_macro_v, time_macro);
            if (first_iteration == true)
                initial_rhs_norm = system_rhs_macro.l2_norm();
            const unsigned int n_iterations = solver.solve_time_step_size_v(system_rhs_macro, system_matrix_macro, solution_macro_v_update,constraints_macro);
            if(n_iterations == 0 || n_iterations == 1 || counter == 6)
            {
                break;
            }
            solution_macro_v += solution_macro_v_update;
            if (first_iteration == true)
                std::cout << "    "  << n_iterations;
            else
                std::cout << '+' << n_iterations;
            first_iteration = false;
            ++counter;
        }
        while (solution_macro_v_update.l2_norm() > 1e-6 );

        std::cout << std::endl;
        std::cout << "     CG iterations per nonlinear step for v." << std::endl;

        system_rhs_macro = 0;
        system_rhs_macro.add(time_step_size*theta , solution_macro_v); // tau theta V^n
        system_rhs_macro.add(time_step_size*(1.-theta) , old_solution_macro_v); // tau (1-theta)  V^n-1
        system_rhs_macro +=  old_solution_macro_u; // U^n-1
        solution_macro_u = system_rhs_macro;

    }

    template <int dim>
    void impMidpoint<dim>::compute_nl_matrix(const Vector<double> &data, SparseMatrix<double> &nl_matrix) 
    {
        QGauss<dim>   quadrature_formula(fe_macro.degree + 1);
        FEValues<dim> fe_values(fe_macro,
                                quadrature_formula,
                                update_values | update_JxW_values |
                                update_quadrature_points);
    
        const unsigned int dofs_per_cell = fe_macro.n_dofs_per_cell();
        const unsigned int n_q_points    = quadrature_formula.size();
    
        FullMatrix<double>                   local_nl_matrix(dofs_per_cell, dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::vector<double>                  data_values(n_q_points);

        for (const auto &cell : dof_handler_macro.active_cell_iterators())
        {
            local_nl_matrix = 0;
            double exp = degree_nl_v - 1.;
            fe_values.reinit(cell);
            fe_values.get_function_values(data, data_values);
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
            double data_abs = std::abs( data_values[q_point]);
            int data_sgn = (data_values[q_point] > 0) - (data_values[q_point] < 0);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                local_nl_matrix(i, j) +=
                    (data_sgn*degree_nl_v*std::pow(data_abs + 0.0001, exp) * // sgn(u)*degree_nl_v*|u + alpha|^(degree_nl_v -1) 
                            fe_values.shape_value(i, q_point) *
                            fe_values.shape_value(j, q_point) * fe_values.JxW(q_point));
            }
            cell->get_dof_indices(local_dof_indices);
            constraints_macro.distribute_local_to_global(local_nl_matrix,
                                                            local_dof_indices,
                                                            nl_matrix);
        }
    }

    template <int dim>
    void impMidpoint<dim>::pre_assemble_system_V(const Vector<double> old_solution_macro_u, const Vector<double> old_solution_macro_v, double time_macro)
    {
        mass_matrix.vmult(pre_system_rhs_macro, old_solution_macro_v); // M*V^n -1
        pre_system_rhs_macro *= -1;
        homogeneous_matrix.vmult(tmp, old_solution_macro_u);
        pre_system_rhs_macro.add(time_step_size, tmp); //  tau_n*A*U^n-1
        homogeneous_matrix.vmult(tmp, old_solution_macro_v); 
        pre_system_rhs_macro.add((1. - theta) * theta *time_step_size *time_step_size, tmp); //  theta*(1-theta)*tau_n^2 *A*v^(n-1)
        if(damped){
            damping_matrix.vmult(tmp,old_solution_macro_v);
            pre_system_rhs_macro.add((1. - theta) *time_step_size, tmp); //  (1-theta)*tau_n B*v^(n-1) 
        }

        rhs_function.set_time(time_macro - time_step_size);
        VectorTools::create_right_hand_side(dof_handler_macro,
                                        QGauss<dim>(fe_macro.degree + 1),
                                        rhs_function,
                                        tmp);
        forcing_terms = tmp; // F^n
        forcing_terms *= -time_step_size * (1. - theta); 
        if (nonlinear) 
        {
            this->compute_nl_term(old_solution_macro_v, tmp);
            forcing_terms.add( time_step_size * (1. - theta),tmp);
        }
        pre_system_rhs_macro += forcing_terms;
        // vereinfachtes NF
        system_matrix_macro.copy_from(mass_matrix);
        system_matrix_macro.add(theta * theta *time_step_size *time_step_size, homogeneous_matrix);  //(M + L +tau_n^2 theta^2 A)
        if (damped)
            system_matrix_macro.add(theta *time_step_size, damping_matrix);  //(M + tau_n^2 theta^2 A + tau_n*theta B)        
        if (nonlinear)
        {
            tmp_matrix = 0;
            compute_nl_matrix(old_solution_macro_v, tmp_matrix);
            system_matrix_macro.add(theta *time_step_size,tmp_matrix); //(M + tau_n^2 theta^2 A + tau_n *theta *nonlinear^n(v_l^n))
        }

    }


    template <int dim>
    void impMidpoint<dim>::assemble_system_V_new( Vector<double>& solution_macro_v, double time_macro)                                                                     
    {
        system_rhs_macro = pre_system_rhs_macro;
        rhs_function.set_time(time_macro);
        VectorTools::create_right_hand_side(dof_handler_macro,
                                            QGauss<dim>(fe_macro.degree + 1),
                                            rhs_function,
                                            tmp);
        forcing_terms = tmp; // F^n
        forcing_terms *= -time_step_size * theta; 
        if (nonlinear)
        {
            this->compute_nl_term(solution_macro_v, tmp); 
            forcing_terms.add(time_step_size * theta,tmp);
        }
        system_rhs_macro += forcing_terms; //  -[tau_n *theta * (F^n + nonlinear^n(v_l^n)) + tau_n *(1-theta) * (F^n-1 + nonlinear^n-1(v^n-1))]
        mass_matrix.vmult(tmp, solution_macro_v);
        system_rhs_macro += tmp;
        homogeneous_matrix.vmult(tmp, solution_macro_v);
        system_rhs_macro.add(time_step_size *time_step_size * theta*theta, tmp); //(M + L + tau_n^2 theta^2 A)

        if(damped)
        {
            damping_matrix.vmult(tmp, solution_macro_v); 
            system_rhs_macro.add(time_step_size *theta, tmp); //(M + tau_n^2 theta^2 A + tau_n*theta B)
        }
        system_rhs_macro *= -1.;
        // normales Newton Verfahren
/*         system_matrix_macro.copy_from(mass_matrix);
        system_matrix_macro.add(theta * theta *time_step_size *time_step_size, homogeneous_matrix);  //(M + L +tau_n^2 theta^2 A)
        if (damped)
            system_matrix_macro.add(theta *time_step_size* beta_value, laplace_matrix);  //(M + tau_n^2 theta^2 A + tau_n*theta B)        
        if (nonlinear)
        {
            tmp_matrix = 0;
            compute_nl_matrix(solution_macro_v, tmp_matrix);
            system_matrix_macro.add(theta *time_step_size,tmp_matrix); //(M + tau_n^2 theta^2 A + tau_n *theta *nonlinear^n(v_l^n))
        } */
        this->apply_Vboundary_condition(system_matrix_macro,system_rhs_macro, solution_macro_v);
        
    }

    /// Save update_matrix and old_step_matrix 
    /// noch nicht entgültig getestet
    ///

    template<int dim>
    void impMidpoint<dim>::integrate_step_alternative(Vector<double>& solution_macro_u,Vector<double>& solution_macro_v,const Vector<double>& old_solution_macro_u,const Vector<double>& old_solution_macro_v,const double time_macro)
    {
        initial_rhs_norm = 0.;
        int counter = 0;
        first_iteration  = true;
        pre_clc_system_V( old_solution_macro_u, old_solution_macro_v, time_macro);
        do
        {
            assemble_clc_V(solution_macro_v, time_macro);
            if (first_iteration == true)
                initial_rhs_norm = system_rhs_macro.l2_norm();
            const unsigned int n_iterations = solver.solve_time_step_size_v(system_rhs_macro, system_matrix_macro, solution_macro_v_update,constraints_macro);
            if(n_iterations == 0 || n_iterations == 1 || counter == 6)
            {
                break;
            }
            solution_macro_v += solution_macro_v_update;
            if (first_iteration == true)
                std::cout << "    "  << n_iterations;
            else
                std::cout << '+' << n_iterations;
            first_iteration = false;
            ++counter;
        }
        while (solution_macro_v_update.l2_norm() > 1e-6 );

        std::cout << std::endl;
        std::cout << "     CG iterations per nonlinear step for v." << std::endl;

        system_rhs_macro = 0;
        system_rhs_macro.add(time_step_size*theta , solution_macro_v); // tau theta V^n
        system_rhs_macro.add(time_step_size*(1.-theta) , old_solution_macro_v); // tau (1-theta)  V^n-1
        system_rhs_macro +=  old_solution_macro_u; // U^n-1
        solution_macro_u = system_rhs_macro;

    }

    template<int dim>
    void impMidpoint<dim>::assemble_system_V(const double time_macro)  // one has to add this before the integration step outside the loop
    {

        update_matrix.copy_from(mass_matrix);
        update_matrix.add(theta * theta *time_step_size *time_step_size, homogeneous_matrix);  //(M + L +tau_n^2 theta^2 A)
        if (damped)
            update_matrix.add(theta *time_step_size, damping_matrix);
        
        old_step_matrix.copy_from(mass_matrix);
        old_step_matrix.add(-theta * theta *time_step_size *time_step_size, homogeneous_matrix);  //(M + L -tau_n^2 theta^2 A)
        if (damped)
            old_step_matrix.add(-theta *time_step_size, damping_matrix);
        
    }

    template <int dim>
    void impMidpoint<dim>::pre_clc_system_V( const Vector<double> old_solution_macro_u, const Vector<double> old_solution_macro_v, double time_macro)
    {
        old_step_matrix.vmult(pre_system_rhs_macro, old_solution_macro_v); // old_step_matrix*V^n -1
        pre_system_rhs_macro *= -1;

        homogeneous_matrix.vmult(tmp, old_solution_macro_u);
        pre_system_rhs_macro.add(time_step_size, tmp); //  tau_n*A*U^n-1

        rhs_function.set_time(time_macro - time_step_size);
        VectorTools::create_right_hand_side(dof_handler_macro,
                                        QGauss<dim>(fe_macro.degree + 1),
                                        rhs_function,
                                        forcing_terms);

        forcing_terms *= -time_step_size * (1. - theta); // F^n-1

        pre_system_rhs_macro += forcing_terms; 


        if (nonlinear) 
        {
            this->compute_nl_term(old_solution_macro_v, tmp);
            pre_system_rhs_macro.add( time_step_size * (1. - theta),tmp);
        }

        // vereinfachtes NF
        system_matrix_macro.copy_from(update_matrix);
        if (nonlinear)
        {
            tmp_matrix = 0;
            compute_nl_matrix(old_solution_macro_v, tmp_matrix);
            system_matrix_macro.add(theta *time_step_size,tmp_matrix); //(M + tau_n^2 theta^2 A + tau_n *theta *nonlinear^n(v_l^n))
        }
    }

    template <int dim>
    void impMidpoint<dim>::assemble_clc_V( Vector<double>& solution_macro_v, double time_macro)                                                                     
    {
        system_rhs_macro = pre_system_rhs_macro;
        rhs_function.set_time(time_macro);
        VectorTools::create_right_hand_side(dof_handler_macro,
                                            QGauss<dim>(fe_macro.degree + 1),
                                            rhs_function,
                                            tmp);
        forcing_terms = tmp; // F^n
        forcing_terms *= -time_step_size * theta; 
        if (nonlinear)
        {
            this->compute_nl_term(solution_macro_v, tmp); 
            forcing_terms.add(time_step_size * theta,tmp);
        }
        system_rhs_macro += forcing_terms; //  -[tau_n *theta * (F^n + nonlinear^n(v_l^n)) + tau_n *(1-theta) * (F^n-1 + nonlinear^n-1(v^n-1))]
        update_matrix.vmult(tmp, solution_macro_v);
        system_rhs_macro += tmp;
        system_rhs_macro *= -1.;
        // normales Newton Verfahren
/*         system_matrix_macro.copy_from(mass_matrix);
        system_matrix_macro.add(theta * theta *time_step_size *time_step_size, homogeneous_matrix);  //(M + L +tau_n^2 theta^2 A)
        if (damped)
            system_matrix_macro.add(theta *time_step_size* beta_value, laplace_matrix);  //(M + tau_n^2 theta^2 A + tau_n*theta B)
            system_matrix_macro.add(theta *time_step_size* beta_value, laplace_matrix);  //(M + tau_n^2 theta^2 A + tau_n*theta B)

        }
            system_matrix_macro.add(theta *time_step_size* beta_value, laplace_matrix);  //(M + tau_n^2 theta^2 A + tau_n*theta B)        

        }
        if (nonlinear)
        {
            tmp_matrix = 0;
            compute_nl_matrix(solution_macro_v, tmp_matrix);
            system_matrix_macro.add(theta *time_step_size,tmp_matrix); //(M + tau_n^2 theta^2 A + tau_n *theta *nonlinear^n(v_l^n))
        } */
        this->apply_Vboundary_condition(system_matrix_macro,system_rhs_macro, solution_macro_v);
        
    }

    ///
    ///
    ///

    template class impMidpoint<1>;
    template class impMidpoint<2>;
}
}
