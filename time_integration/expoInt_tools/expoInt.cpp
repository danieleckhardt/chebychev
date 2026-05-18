#include "expoInt.h"
#include <variant>
#include <sstream>
#include <type_traits>

namespace MavesS {
namespace TimeIntegration
{   

template<int dim>
ExpoInt<dim>::ExpoInt(FE_Q<dim>&  fe_macro, DoFHandler<dim>&  dof_handler_macro,AffineConstraints<double>& constraints_macro,  
                        SparseMatrix<double>& homogeneous_matrix,SparseMatrix<double>& damping_matrix, SparseMatrix<double>& mass_matrix,
                        double& time_step_size, const double & degree_nl_v, bool info_ellipse)
    :
    TimeIntegrator<dim>( fe_macro, dof_handler_macro, constraints_macro, degree_nl_v)
    ,info_ellipse(info_ellipse)
    ,time_step_size(time_step_size)
    ,mass_matrix(mass_matrix)
    ,homogeneous_matrix(homogeneous_matrix)
    ,damping_matrix(damping_matrix)
    ,cheb_approx(10, ellipse_opt, time_step_size,homogeneous_matrix,mass_matrix) 
    ,calc_eigen(eigenvalues, eigenvectors)
    ,arnoldi_krylov(homogeneous_matrix, mass_matrix ,krylov_vectors_u,krylov_vectors_v, time_step_size)
    {}


template<int dim>
void ExpoInt<dim>::setup_system_macro_space()
{
    M_inv = std::make_unique<Types::InverseMatrix<dim,double>>(mass_matrix,dof_handler_macro);
    find_ellipse = std::make_unique<FindEllipse<dim>>(mass_matrix, homogeneous_matrix, damping_matrix, dof_handler_macro, info_ellipse);
    tmp_vec.reinit(homogeneous_matrix.n());
    approx_result.reinit(dof_handler_macro.n_dofs(),2);
}

template<int dim>
void ExpoInt<dim>::setup_system_macro_time( double new_timestep)
{
    time_step_size = new_timestep;
}

template<int dim>
void ExpoInt<dim>::store_info_ellipse(std::string filename)
{
    using InfoVariant =
    std::variant<
        int,
        double,
        std::vector<double>,
        std::vector<std::vector<
            std::variant<std::complex<double>, double, int>
        >>
    >;

    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    auto variant_to_string = [&](const InfoVariant &var)->std::string{
        std::ostringstream ss;
        std::visit([&](auto&& val){
            using T = std::decay_t<decltype(val)>;

            if constexpr (std::is_same_v<T,int> || std::is_same_v<T,double>) {
                ss << val;
            }
            else if constexpr (std::is_same_v<T,std::vector<double>>) {
                ss << '[';
                for (size_t i = 0; i < val.size(); ++i) {
                    if (i) ss << ',';
                    ss << val[i];
                }
                ss << ']';
            }
            else {
                // vector<vector<variant<complex<double>, double, int>>>
                ss << '[';
                for (size_t i = 0; i < val.size(); ++i) {
                    if (i) ss << ',';
                    ss << '(';
                    for (size_t j = 0; j < val[i].size(); ++j) {
                        if (j) ss << ',';
                        std::visit([&](auto&& inner){
                            ss << inner;
                        }, val[i][j]);
                    }
                    ss << ')';
                }
                ss << ']';
            }
        }, var);
        return ss.str();
    };

    for (const auto &step : find_ellipse->info_steps) {
        out << variant_to_string(step[0]);
        if(step.size() >= 2 ){
            out << ";";
            out << variant_to_string(step[1]);
        }
        if(step.size() == 4){
            out << ";";
            out << variant_to_string(step[2]) << ";";
            out << variant_to_string(step[3]);
        }
        out << '\n';
    }
    for ( const auto &p : find_ellipse->eigenvalues_pub){
        out << p << ";";
    }
    out << '\n';
    std::vector<double> vtemp = {a_out, b_out, c_out};
    out << variant_to_string(vtemp);
}



///////////////////////////
/////////////////////////// 
///////////////////////////

template< int dim>
std::vector<int> ExpoInt<dim>::find_opt_ellipse_prototype(const FullMatrix<double>& A){
    // Iif A is empty, create it on the fly
    if (A.n() > 0 && A.m() > 0) {
        ellipse_config = find_ellipse->prototype(time_step_size, A);
    }
    else {
        ellipse_config = find_ellipse->prototype(time_step_size);
    }
    find_ellipse->get_ellipse(ellipse_opt);
    ellipse_opt.get_radx(a_out);
    ellipse_opt.get_rady(b_out);
    ellipse_opt.get_center(c_out);
    return ellipse_config;
}

template<int dim>
std::vector<std::complex<double>> ExpoInt<dim>::find_opt_ellipse(double tol ,Types::BlockMatrix<dim>* block_matrix_ptr) {
    std::cout<< "Search for optimal ellipse!" << std::endl;
    Types::BlockMatrix<dim> local_block_matrix(homogeneous_matrix, damping_matrix, mass_matrix, dof_handler_macro, time_step_size,false);
    // if block_matrix_ptr is provided, use it; otherwise, use the local_block_matrix
    Types::BlockMatrix<dim>& block_matrix = block_matrix_ptr ? *block_matrix_ptr : local_block_matrix;
    std::vector<std::complex<double>> points{std::complex<double>(0., 0.0)};
    std::cout << " Calculate largest eigenvalue..." << std::endl;
    calc_eigen.calc_max_eig(block_matrix, 4, 1e-10); 
    std::cout << " Start with point_0 = 0. and point_1 = " << eigenvalues[0] << std::endl;
    points.push_back(eigenvalues[0]);
    find_ellipse->chebyshev_arnoldi(1,block_matrix,points,tol,time_step_size);
    find_ellipse->get_ellipse(ellipse_opt);
    return points;
}

template<int dim>
int ExpoInt<dim>::estimate_degree_phi_k(int phi_k, double tol, int init_degree){
    find_ellipse->set_ellipse(ellipse_opt);
    degree =  find_ellipse->estimate_degree_max(phi_k,tol,init_degree);
    return degree;
}

template<int dim>
int ExpoInt<dim>::estimate_degree(std::function<std::complex<double>(std::complex<double>)> func,  double tol, int init_degree){
    find_ellipse->set_ellipse(ellipse_opt);
    degree = find_ellipse->estimate_degree_max(func,tol,init_degree);
    return degree;
}

template<int dim>
void ExpoInt<dim>::matrix_func_cheb(BMatrixVariant Matrix_mult, std::function<std::complex<double>(std::complex<double>)> func, Vector<double> &u, Vector<double> &v, double tol, int k_phi){
    cheb_approx.set_degree(degree+1);

    for(unsigned int j = 0; j < u.size(); ++j){
            approx_result(j,0) = u(j);
            approx_result(j,1) = v(j);
    }
    // Run Chebyshev approximation
    cheb_approx.matrix_run(Matrix_mult, func, approx_result, std::nullopt, tol, k_phi);
    cheb_approx.get_count_mv(mv_count_per_step);
    //
    for(unsigned int j = 0; j < u.size(); ++j){
            u(j) = approx_result(j,0).real();
            v(j) = approx_result(j,1).real();
    }
}

///////////////////////////
///////////////////////////
///////////////////////////

template<int dim>
void ExpoInt<dim>::set_max_iteration_krylov(int max_iter)
{
    max_iterations = max_iter;
    arnoldi_krylov.set_max_iterations(max_iterations);
}

template<int dim>
bool ExpoInt<dim>::matrix_func_poly_krylov(BMatrixVariant Matrix_mult,std::function<std::complex<double>(std::complex<double>)> func, Vector<double> &u, Vector<double> &v, const double tol) // approximate func([ 0 M^-1 ] [M^-1 A  M^-1 B])[u  v]
{
    krylov_vectors_u[0] = u;
    krylov_vectors_v[0] = v;
    
    set_max_iteration_krylov(max_iterations);
    bool konv = arnoldi_krylov.run_poly(Matrix_mult,func,tol);
    norm_value_krylov = arnoldi_krylov.get_norm(); 
    arnoldi_krylov.get_matrix_funcxe1_krylov(matrix_funcxe1_krylov); 
    iterations_needed = arnoldi_krylov.get_iterations();
    arnoldi_krylov.get_count_mv(mv_count_per_step);
        
    //Evaluate approximation to  phi(A)
    u = 0;
    v = 0;
    for(int i=0; i<iterations_needed; ++i){
        u.add(matrix_funcxe1_krylov[i]*norm_value_krylov, krylov_vectors_u[i]); // + krylov_vector * matrix_funcxe1_krylov[i]
        v.add(matrix_funcxe1_krylov[i]*norm_value_krylov, krylov_vectors_v[i]);  
    }
    if(konv == false){
            std::cout << "No Convergence!" << std::endl;
            return false;
    }

    return true;
}

template<int dim>
bool ExpoInt<dim>::matrix_func_rat_krylov(BMatrixVariant Matrix_mult,BMatrixVariant Matrix_mult_inv,std::function<std::complex<double>(std::complex<double>)> func, Vector<double> &u, Vector<double> &v, const double tol,  const double shift_parameter) // approximate func([ 0 M^-1 ] [M^-1 A  M^-1 B])[u  v]
{
    krylov_vectors_u[0] = u;
    krylov_vectors_v[0] = v;
    
    set_max_iteration_krylov(max_iterations);
    arnoldi_krylov.set_weighted_sp(true);
    arnoldi_krylov.set_rat_shift(shift_parameter);
    bool konv = arnoldi_krylov.run_rational(Matrix_mult,Matrix_mult_inv,func,tol);
    norm_value_krylov = arnoldi_krylov.get_norm(); 
    arnoldi_krylov.get_matrix_funcxe1_krylov(matrix_funcxe1_krylov); 
    iterations_needed = arnoldi_krylov.get_iterations();
    arnoldi_krylov.get_count_mv(mv_count_per_step);
        
    //Evaluate approximation to  phi(A)
    u = 0;
    v = 0;
    for(int i=0; i<iterations_needed; ++i){
        u.add(matrix_funcxe1_krylov[i]*norm_value_krylov, krylov_vectors_u[i]); // + krylov_vector * matrix_funcxe1_krylov[i]
        v.add(matrix_funcxe1_krylov[i]*norm_value_krylov, krylov_vectors_v[i]); 
    }
    // for (int i = 0 ; i < krylov_vectors_u[0].size() ; i++){
    //     std::cout << " u"  << u[i];
    // }
    if(konv == false){
            std::cout << "No Convergence!" ;
            return false;
    }

    return true;
}

template class ExpoInt<1>;
template class ExpoInt<2>;

}
}
