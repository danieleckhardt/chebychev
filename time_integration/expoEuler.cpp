#include "expoEuler.h"

namespace MavesS {
namespace TimeIntegration
{   

template<int dim> 
ExpoEuler<dim>::ExpoEuler(FE_Q<dim>& fe_macro, DoFHandler<dim>& dof_handler_macro, AffineConstraints<double>& constraints_macro, 
                            SparseMatrix<double>& homogeneous_matrix, SparseMatrix<double>& damping_matrix, SparseMatrix<double>& mass_matrix, std::string method_calc_mfunc,
                            double& time_step_size, const double & degree_nl_v, const int example, const bool damped, const bool nonlinear)
    :
    ExpoInt<dim>(fe_macro, dof_handler_macro, constraints_macro,homogeneous_matrix,damping_matrix,mass_matrix,time_step_size, degree_nl_v)
    ,damped(damped)
    ,nonlinear(nonlinear)
    ,method_calc_mfunc(method_calc_mfunc)
    ,rhs_function(example)
    {}

template<int dim>
void ExpoEuler<dim>::setup_system_macro_space(){
    ExpoInt<dim>::setup_system_macro_space();

    bmatrix =  std::make_unique<Types::BlockMatrix<dim>>(homogeneous_matrix, damping_matrix, mass_matrix, dof_handler_macro, time_step_size,true,true); // set last false if damping matrix ist not symmetric!
    double tol_inv = tol*1e-3;
    phi1_lambda = [this](std::complex<double> x){ return func_matrix.phi1(x); };
    if(method_calc_mfunc == "cheb" ){
        bmatrix_vmult = [this,tol_inv](FullMatrix<std::complex<double>> & Ma) { return this->bmatrix->vmultBlock(Ma, tol_inv ); };
    }else if (method_calc_mfunc == "krylov"){
        bmatrix_vmult = [this,tol_inv](Vector<double>& a ,Vector<double> & b) { return this->bmatrix->vmult(a,b,tol_inv); };
    }else if (method_calc_mfunc == "rat_krylov"){
        bmatrix->set_rat_shift(rat_shift*time_step_size);
        bmatrix->set_factor(time_step_size);
        bmatrix_vmult = [this,tol_inv](Vector<double>& a ,Vector<double> & b) { return this->bmatrix->vmult_shift(a,b,tol_inv); };
        bmatrix_vmult_inv = [this,tol_inv](Vector<double>& a ,Vector<double> & b) { return  this->bmatrix->vmult_rat(a,b,tol_inv); };
    }
    else{
        std::cerr << "choose krylov, rat_krylov, chebyshev" << std::endl;
    }
}

template<int dim>
void ExpoEuler<dim>::setup_system_macro_time( double new_timestep){
    ExpoInt<dim>::setup_system_macro_time( new_timestep);
    bmatrix->set_factor(new_timestep);
}

template<int dim>
void ExpoEuler<dim>::integrate_step(Vector<double>& solution_macro_u,Vector<double>& solution_macro_v, const Vector<double>&  old_solution_macro_u,const Vector<double>&  old_solution_macro_v, const double time_macro){
    // Calculate f - Sx
    AssertThrow(solution_macro_u.size() == homogeneous_matrix.m(),
            ExcMessage("solution_macro_u not properly initialized"));
    AssertThrow(solution_macro_v.size() == damping_matrix.m(),
            ExcMessage("solution_macro_v not properly initialized"));

    solution_macro_u = old_solution_macro_v; 
    //
    homogeneous_matrix.vmult(solution_macro_v,old_solution_macro_u);
    if(damped){
        damping_matrix.vmult(tmp_vec,old_solution_macro_v);
        solution_macro_v += tmp_vec;
        mv_count += 1; 

    }
    mv_count += 1; 
    //
    rhs_function.set_time(time_macro);
    VectorTools::create_right_hand_side(dof_handler_macro,
                                        QGauss<dim>(fe_macro.degree + 1),
                                        rhs_function,
                                        tmp_vec);

    solution_macro_v -= tmp_vec;
    if (nonlinear) {
        this->compute_nl_term(old_solution_macro_v, tmp_vec); 
        solution_macro_v += tmp_vec;
    }
    solution_macro_v *= -1;
    mv_count += M_inv->vmult(solution_macro_v,solution_macro_v,tol);
    //
    if (method_calc_mfunc == "cheb") {

        this->matrix_func_cheb(bmatrix_vmult, phi1_lambda, solution_macro_u, solution_macro_v, tol, 1);
        cheb_approx.get_degree(degree_cheb);
        int mv_count_cheb;
        cheb_approx.get_count_mv(mv_count_cheb);
        mv_count += mv_count_cheb;

    } else if (method_calc_mfunc == "krylov") {

        this->matrix_func_poly_krylov(bmatrix_vmult,phi1_lambda, solution_macro_u, solution_macro_v, tol);
        
        iteration_krylov =  arnoldi_krylov.get_iterations();
        int mv_count_krylov;
        arnoldi_krylov.get_count_mv(mv_count_krylov);
        mv_count += mv_count_krylov;

    } else if (method_calc_mfunc == "rat_krylov") {
        this->matrix_func_rat_krylov(bmatrix_vmult,bmatrix_vmult_inv,phi1_lambda, solution_macro_u, solution_macro_v, tol, rat_shift);
        iteration_krylov =  arnoldi_krylov.get_iterations();
        int mv_count_krylov;
        arnoldi_krylov.get_count_mv(mv_count_krylov);
        mv_count += mv_count_krylov;

    } else {
        AssertThrow(false, ExcMessage("Unknown method for calculating the matrix function"));
    }
    //
    solution_macro_u *= time_step_size;
    solution_macro_v *= time_step_size;

    solution_macro_u += old_solution_macro_u;
    solution_macro_v += old_solution_macro_v;
    
}

template class ExpoEuler<1>;
template class ExpoEuler<2>;
                                        
}
}