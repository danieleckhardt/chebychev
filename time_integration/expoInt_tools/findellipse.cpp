#include "findellipse.h"

namespace MavesS {
namespace TimeIntegration
{   

template<int dim>
FindEllipse<dim>::FindEllipse(SparseMatrix<double>& mass_matrix, SparseMatrix<double>& homogeneous_matrix, SparseMatrix<double>& damping_matrix,  DoFHandler<dim>&  dof_handler, bool store_info)
    :
    info_steps()
    ,max_steps(30)
    ,time_step_size(1.0) 
    ,store_info(store_info)
    ,info_single_step()
    ,ellipse_parameters()
    ,info_step_points()
    ,info_step_single_point()
    ,mass_matrix(mass_matrix)
    ,M_inv(mass_matrix,dof_handler)
    ,homogeneous_matrix(homogeneous_matrix)
    ,damping_matrix(damping_matrix)
    ,dof_handler(dof_handler)
    ,calc_eigen(eigenvalues, eigenvectors)
    ,cheb_approx(10, ellipse, time_step_size,homogeneous_matrix,mass_matrix)
    ,arnoldi_krylov(homogeneous_matrix, mass_matrix, krylov_vectors_u, krylov_vectors_v,time_step_size)
    {}

template<int dim> 
bool FindEllipse<dim>::almost_equal(const std::complex<double>& a, 
                  const std::complex<double>& b, 
                  double eps ) {
    return std::sqrt(std::pow(a.real() - b.real(), 2) + std::pow(a.imag() - b.imag(),2)) < eps;
}

template<int dim> 
void FindEllipse<dim>::create_bd_point() {
    double r_x, r_y, c;
    std::complex<double> foci;
    ellipse.get_radx(r_x);ellipse.get_rady(r_y); 
    ellipse.get_center(c);ellipse.get_ecc(foci);
    if(foci.imag() == 0){
        bd_point = c + r_x;
    }
    else{
        bd_point = c + r_y*1i;
    }
}

template<int dim>
void FindEllipse<dim>::get_start_vectors(
    Vector<std::complex<double>>& start_vec_u,
    Vector<std::complex<double>>& start_vec_v,
    bool orthogonalize)
{
    // --- Random number generator (kept local, but could be static for efficiency)
    // std::random_device rd;
    const unsigned int seed = 80; // Fixed seed for reproducibility
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    const unsigned int n = dof_handler.n_dofs();

    // --- Resize and initialize random vectors and store
    Vector<std::complex<double>> tmpu(n), tmpv(n);
    tmpu = start_vec_krylov_u;
    tmpv = start_vec_krylov_v;
    start_vec_u.reinit(n);
    start_vec_v.reinit(n);

    for (unsigned int i = 0; i < n; ++i)
    {
        start_vec_u[i] = std::complex<double>(dis(gen), dis(gen));
        start_vec_v[i] = std::complex<double>(dis(gen), dis(gen));
    }

    if (!orthogonalize)
        return;

    // --- Apply operators
    Vector<std::complex<double>> Au(n), Mv(n);

    homogeneous_matrix.vmult(Au, tmpu);
    mass_matrix.vmult(Mv, tmpv);

    // --- Compute inner products (Hermitian)
    std::complex<double> num = 0.0;
    std::complex<double> den = 0.0;

    for (unsigned int i = 0; i < n; ++i)
    {
        num += std::conj(start_vec_u[i]) * Au[i];
        num += std::conj(start_vec_v[i]) * Mv[i];

        den += std::conj(tmpu[i]) * Au[i];
        den += std::conj(tmpv[i]) * Mv[i];
    }

    // --- Safety check
    const double norm = std::abs(den);
    if (norm == 0.0)
        return;

    std::complex<double> alpha = num / den;

    // --- Projection step (consistent update)
    for (unsigned int i = 0; i < n; ++i)
    {
        start_vec_u[i] -= alpha * start_vec_u[i];
        start_vec_v[i] -= alpha * start_vec_v[i];
    }
}

template <int dim>
void FindEllipse<dim>::extract_real_part(
    Vector<std::complex<double>>& vec_u,
    Vector<std::complex<double>>& vec_v)
{
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i) {
        vec_u[i] = std::real(vec_u[i]);
        vec_v[i] = std::real(vec_v[i]);
    }
}

template<int dim>
bool FindEllipse<dim>::estimate_degree(int& degree, double tol){
    // estimate degree
    std::cout << "Estimated degree..." << std::endl;
    if (radius_max > (1.0+ tol*100.)){
        double damping = 1./ radius_max;
        degree = std::max(static_cast<int>(std::round(0.5*std::abs(std::log(tol)/std::log(damping)))), degree ) +10 ;
        std::cout << "  ...using that there is still a gap." << std::endl;
    }
    else{
        degree = static_cast<int>(std::round( degree* (1 + 0.1*std::abs(std::log(radius_max/(1.+tol) ) + 1 )))) ;
        std::cout << "  ...near to convergence." << std::endl;

    }
    if (degree < degree_max)
        return false;
    else{
        degree = degree_max;
        return true;
    }
}

template <int dim>
bool FindEllipse<dim>::check_approximate_eigenpair(
    Types::BlockMatrix<dim>& block_matrix,
    const std::complex<double>& lambda,
    const Vector<std::complex<double>>& eigvec_u,
    const Vector<std::complex<double>>& eigvec_v,
    double tol) 
{
    // Step 1: Apply A * v
    Vector<std::complex<double>> Au(eigvec_u.size());
    Vector<std::complex<double>> Av(eigvec_v.size());
    block_matrix.vmult_complex_split_weak_op(Au, Av, eigvec_u, eigvec_v);

    // Step 2: Form λ * v
    Vector<std::complex<double>> lambda_u = eigvec_u;
    Vector<std::complex<double>> lambda_v = eigvec_v;
    homogeneous_matrix.vmult(lambda_u,eigvec_u);
    mass_matrix.vmult(lambda_v,eigvec_v);
    lambda_u *= lambda;
    lambda_v *= lambda;

    // Step 3: Compute residual r = Av - λv
    Vector<std::complex<double>> res_u = Au;
    Vector<std::complex<double>> res_v = Av;
    res_u -= lambda_u;
    res_v -= lambda_v;

    // Step 4: Compute relative residual
    double res = std::sqrt(
        std::pow(res_u.l2_norm(), 2) + std::pow(res_v.l2_norm(), 2));
    double norm_Auv = std::sqrt(
        std::pow(Au.l2_norm(), 2) + std::pow(lambda_u.l2_norm(), 2) + std::pow(Av.l2_norm(), 2) + std::pow(lambda_v.l2_norm(), 2));
    // std::cout << "norm_Auv" << norm_Auv << std::endl;
    relative_residual = res / (norm_Auv + 1e-14);
    if (relative_residual  < global_res && ellipse.radius_p_alterative(lambda) >= ellipse.radius_p_alterative(sorted_eigenvalues[0])){
        // take as new startvector, eig_vec with rad(eig_val) is largest and res is smallest 
        start_vec_krylov_u = eigvec_u;
        start_vec_krylov_v = eigvec_v;
        global_res = relative_residual;
    }

    if (relative_residual < tol) {
        std::cout << "  Accepted λ=" << lambda
                    << " with residual " << relative_residual << " and Radius: " << ellipse.radius_p_alterative(lambda)/ellipse.radius_p_alterative(bd_point) << ". ";
        return true;
    } else {
        std::cout << "  Rejected λ=" << lambda
                    << " with residual " << relative_residual << " and Radius: " << ellipse.radius_p_alterative(lambda)/ellipse.radius_p_alterative(bd_point)<< std::endl;
        
        return false;
    }
}

template <int dim>
void FindEllipse<dim>::build_reduced_matrix(Types::BlockMatrix<dim>& block_matrix, const int iterations_needed){
    const unsigned int n = homogeneous_matrix.n();

    // Step 1: Compute A * V
    std::vector<Vector<std::complex<double>>> tmp_u(iterations_needed, Vector<std::complex<double>>(n));
    std::vector<Vector<std::complex<double>>> tmp_v(iterations_needed, Vector<std::complex<double>>(n));

    for (int i = 0; i < iterations_needed; ++i) {
        block_matrix.vmult_complex_split_weak_op(tmp_u[i], tmp_v[i], krylov_vectors_u[i], krylov_vectors_v[i]);
    }
    // Step 2: Project into Krylov basis -> fill Hm_block_matrix
    // Note: V_m is orthogonal w.r.t. [[A 0][0 M]]
    Hm_block_matrix.reinit(iterations_needed, iterations_needed);
    for (int i = 0; i < iterations_needed; ++i) {
        for (int j = 0; j < iterations_needed; ++j) {
            std::complex<double> val_ij = krylov_vectors_u[i] * tmp_u[j]
                                        + krylov_vectors_v[i] * tmp_v[j];
            Hm_block_matrix(i, j) = val_ij;
        }
    }
}

template <int dim>
void FindEllipse<dim>::reconstruct_and_sort_eigenpairs(const int iterations_needed){
    const unsigned int n = krylov_vectors_u[0].size();

    // Step 1: Reconstruct approximate eigenvectors in full space
    std::vector<Vector<std::complex<double>>> approx_u(iterations_needed, Vector<std::complex<double>>(n));
    std::vector<Vector<std::complex<double>>> approx_v(iterations_needed, Vector<std::complex<double>>(n));

    for (int i = 0; i < iterations_needed; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            std::complex<double> sum_u = 0.0;
            std::complex<double> sum_v = 0.0;
            for (int k = 0; k < iterations_needed; ++k) {
                sum_u += krylov_vectors_u[k][j] * eigenvectors[k][i];
                sum_v += krylov_vectors_v[k][j] * eigenvectors[k][i];
            }
            approx_u[i][j] = sum_u;
            approx_v[i][j] = sum_v;
        }
    }

    // Step 2: Sort indices by ellipse radius
    std::vector<int> indices(eigenvalues.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::stable_sort(indices.begin(), indices.end(), [this](int i1, int i2) {
        return ellipse.radius_p_alterative(eigenvalues[i1]) > ellipse.radius_p_alterative(eigenvalues[i2]);
    });

    // Step 3: Reorder eigenpairs
    sorted_eigenvalues.clear();
    sorted_eigenvectors_u.clear();
    sorted_eigenvectors_v.clear();

    for (int idx : indices) {
        sorted_eigenvalues.push_back(eigenvalues[idx]);
        sorted_eigenvectors_u.push_back(approx_u[idx]);
        sorted_eigenvectors_v.push_back(approx_v[idx]);
    }
    start_vec_krylov_u = sorted_eigenvectors_u[0];
    start_vec_krylov_v = sorted_eigenvectors_v[0];

}

template<int dim>
bool FindEllipse<dim>::estimate_ellipse_chebyshev_arnoldi( std::function<int(Vector<std::complex<double>>&, Vector<std::complex<double>>&)> cheb_n,
                                                            Types::BlockMatrix<dim>& block_matrix,
                                                            std::vector<std::complex<double>>& detected_points,
                                                            double tol_fix,double & tol_variable,
                                                            int steps){
    int found_count = 0;
    bool found_new_points = false;
    // const int n = homogeneous_matrix.n();
    double r_x,r_y,ce;
    std::cout << " "  << std::endl;
    std::cout << "steps: " << steps << std::endl;
    //
    if (store_info){
        info_single_step.clear();
        ellipse_parameters.clear();
        info_step_points.clear();
    }
    //
    
    // Run Arnoldi algorithm
    arnoldi_krylov.run_poly(cheb_n, nullptr, tol_variable);
    int iterations_needed = arnoldi_krylov.get_iterations();

    // Estimate spectral radius
    std::complex<double> maxeig = arnoldi_krylov.get_max_eigenvalue();
    radius_max = std::abs(std::pow(std::abs(maxeig), 1.0 / (degree)));
    //
    if (store_info){
        info_single_step.push_back(degree);
        info_single_step.push_back(radius_max);
    }
    //

    if (radius_max <= 1.0 + tol_fix){
        std::cout << "The maximum radius is " << radius_max << " <= "  << 1.0 + tol_fix << ". (degree =  " << degree << ")."  << std::endl;
        return false;
    } 
    else
        std::cout << "The maximum radius is " << radius_max << " > "  << 1.0 + tol_fix << ". Continue with tol = " << 1.0 + tol_fix  << " and degree =  " << degree << " . " << std::endl;
    
    // Compute (V^T , A @ V)_{H^1 x L^2} : 
    build_reduced_matrix(block_matrix,iterations_needed);
   
    // Compute eigenvalues and eigenvectors of the reduced H matrix
    calc_eigen.calc_all_eig(Hm_block_matrix, iterations_needed);

    // Reconstruct approximate eigenvectors
    reconstruct_and_sort_eigenpairs(iterations_needed);
    
    // Check approximate eigenpairs
    std::cout << "Checking approximate eigenpairs..." << std::endl;
    for (size_t count = 0; count < sorted_eigenvalues.size(); ++count) {
        const std::complex<double>& lambda = sorted_eigenvalues[count];
        double lambda_re = lambda.real();
        double lambda_imag = lambda.imag();
        
        if ( ellipse.radius_p_alterative(lambda) <= 1.0 || lambda.imag() < 0. ) {
            continue; // Skip eigenvalues with negative imaginary part or inside ellipse
        }
        //
        if (store_info){
            info_step_single_point.clear();
            info_step_single_point.push_back(lambda_re);
            info_step_single_point.push_back(lambda_imag);
        }
        //
        if (check_approximate_eigenpair(block_matrix, lambda,sorted_eigenvectors_u[count],sorted_eigenvectors_v[count],tol_fix)){
            auto it = std::find_if(detected_points.begin(), detected_points.end(), [&](const std::complex<double>& c){ return almost_equal(c, lambda, tol_fix);});
            if (it == detected_points.end()) {
                detected_points.push_back(lambda);
                // detected_vectors.push_back({sorted_eigenvectors_u[count],
                //                             sorted_eigenvectors_v[count]});
                std::cout << "New eigenpair!" << " (" << found_count  <<  ". in this loop)." <<  std::endl;
                found_count++;
                found_new_points = true;
                if(store_info){
                    info_step_single_point.push_back(relative_residual);
                    info_step_single_point.push_back(1);
                    info_step_points.push_back(info_step_single_point);
                }
            }
            else{
                std::cout << "Eigenpair already detected." << std::endl;
                if(store_info){
                    info_step_single_point.push_back(relative_residual);
                    info_step_single_point.push_back(-1);
                    info_step_points.push_back(info_step_single_point);
                }
            }
        }
        else{
            if (store_info){
                info_step_single_point.push_back(relative_residual);
                info_step_single_point.push_back(0);
                info_step_points.push_back(info_step_single_point);
            }
        }
    }
    global_res = 1.;
    // If no new ellipse, use smaller tol 
    ellipse_config = ellipse.create_opt_ellipse(detected_points);
    ellipse.get_radx(r_x);
    ellipse.get_rady(r_y);
    ellipse.get_center(ce);
    if (store_info){
        ellipse_parameters.push_back(r_x);
        ellipse_parameters.push_back(r_y);
        ellipse_parameters.push_back(ce);
        info_single_step.push_back(ellipse_parameters);
        info_single_step.push_back(info_step_points);
    }

    if (found_new_points){
        tol_variable = tol_fix;
    }
    else{
        if (!degree_max_reached){
            // degree += 10;
        }
        else{
            tol_variable *= 0.1;
            degree_max_reached = false;
            // degree_max += 5; 
            std::cout << " Maximal degree has been reached. Choose smaller tol=" << tol_variable << " for Arnoldi process. " << std::endl;
        }
        ellipse.get_rady(r_y);
        ellipse.set_rady_with_fix_center_and_ecc(r_y*0.95); // better convergence, keep foci and center fixed to converge against wanted eigenvalues
    }
    create_bd_point();
    return true;
}

////////////////////////////////
/*
    Public Methods
*/
///////////////////////////////

template<int dim>
int FindEllipse<dim>::estimate_degree_max(std::function<std::complex<double>(std::complex<double>)> func,  double tol, int init_degree){
    // estimate degree
    int estimated_degree = 0;
    cheb_approx.set_degree(init_degree);
    cheb_approx.estimate_degree_ellipse_by_error(func,tol);
    cheb_approx.get_degree(estimated_degree);
    std::cout << "  Set max degree : " << estimated_degree << std::endl;
    return estimated_degree  ; 
}

template<int dim>
int FindEllipse<dim>::estimate_degree_max(int phi_k, double tol, int init_degree){
    // estimate degree
    // int estimated_degree = 0;
    // cheb_approx.set_degree(init_degree);
    // cheb_approx.estimate_degree_ellipse_by_estimate(phi_k,tol);
    // cheb_approx.get_degree(estimated_degree);
    int degree = 1000;
    cheb_approx.set_degree(degree);
    std::cout << "  Set max degree : " << degree << std::endl;
    return degree  ; 
}

template< int dim> 
std::vector<int> FindEllipse<dim>::prototype(double time_step_size){
    // calculate eigenvalues
    size = dof_handler.n_dofs();
    FullMatrix<double> A_full_matrix;
    const IdentityMatrix id_matrix(size);
    FullMatrix<double> identity_m = id_matrix; 

    A_full_matrix.reinit(2*size,2*size);
    std::vector<int> first_index_set  = linspace(0,size,1);
    std::vector<int> second_index_set = linspace(size,2*size,1);

    identity_m.scatter_matrix_to(first_index_set,second_index_set,A_full_matrix);

    FullMatrix<double> homogeneous_matrix_full(size,size);
    homogeneous_matrix_full.copy_from(homogeneous_matrix);
    homogeneous_matrix_full *= -1.;
    M_inv.mmult(homogeneous_matrix_full,homogeneous_matrix_full,1e-10);
    homogeneous_matrix_full.scatter_matrix_to(second_index_set,first_index_set,A_full_matrix);

    FullMatrix<double> damping_matrix_full(size,size);
    damping_matrix_full.copy_from(damping_matrix);
    damping_matrix_full *= -1.;
    M_inv.mmult(damping_matrix_full,damping_matrix_full,1e-10);
    damping_matrix_full.scatter_matrix_to(second_index_set,second_index_set,A_full_matrix);
    A_full_matrix *= time_step_size; // scale matrix with time step size
    calc_eigen.calc_all_eig(A_full_matrix, 2*size);

    return ellipse.create_opt_ellipse(eigenvalues);
}

template< int dim> 
std::vector<int> FindEllipse<dim>::prototype(double time_step_size, const FullMatrix<double>& A){
    // calculate eigenvalues
    int size = A.n();
    calc_eigen.calc_all_eig(A, size);
    
    std::vector<std::complex<double>> ellipse_points = {0., eigenvalues[0]};
    ellipse_config = ellipse.create_opt_ellipse(ellipse_points);
    // double max_rad = 0.;
    for (unsigned long i = 0; i < eigenvalues.size(); ++i) {
        eigenvalues[i] *= time_step_size;
        // std::cout << "Eigenvalue " << i << ": " << eigenvalues[i] << " Radius: " << ellipse.radius_p(eigenvalues[i]) << "; " ;
        // if (max_rad < ellipse.radius_p(eigenvalues[i])){
        //     max_rad = ellipse.radius_p(eigenvalues[i]);
        // }
    }
    eigenvalues_pub = eigenvalues;
    // std::cout << "max_rad:" << max_rad;
    return ellipse.create_opt_ellipse(eigenvalues);

}

template<int dim>
void FindEllipse<dim>::chebyshev_arnoldi(int phi_k, Types::BlockMatrix<dim>& block_matrix,  std::vector<std::complex<double>>& detected_points, double tol_fix, double time_step_size) {
    double radius_tempx,radius_tempy,center;
    std::complex<double> foci;
    unsigned long points_size = detected_points.size();
    max_iter_arnoldi = std::min(200,2*block_matrix.get_size());
    arnoldi_krylov.set_time_step_size(time_step_size);
    arnoldi_krylov.set_max_iterations(max_iter_arnoldi);

    // Initialize Matrix-Vector operation
    BMatrixVariant Matrix_mult = [&block_matrix](FullMatrix<std::complex<double>>& block_vec) {
        return block_matrix.vmultBlock(block_vec, 1e-10);
    };

    get_start_vectors(krylov_vectors_u[0], krylov_vectors_v[0],false);
    double tol_variable = 1e-2;

    // Generate initial ellipse from eigenvalues
    ellipse_config = ellipse.create_opt_ellipse(detected_points);
    ellipse.get_ecc(foci);
    degree_max = estimate_degree_max(phi_k,tol_fix, 10) ;
    degree = std::min( 2*static_cast<int>(std::sqrt(std::abs(foci))) + 10 , degree_max); // initial degree
    cheb_approx.set_chebyshev_tn_basis(degree);
    create_bd_point();

    // matrix-vector operation using Chebyshev polynomial
    block_vec.reinit(dof_handler.n_dofs(),2);
    std::function<int(Vector<std::complex<double>>&, Vector<std::complex<double>>&)> cheb_n =
        [this, Matrix_mult](Vector<std::complex<double>>& b_1, Vector<std::complex<double>>& b_2) {
            for(unsigned int j = 0; j < b_1.size(); ++j){
                    this->block_vec(j,0) = b_1(j);
                    this->block_vec(j,1) = b_2(j);
            }
            cheb_approx.matrix_run(Matrix_mult, nullptr, this->block_vec, this->bd_point);
            for(unsigned int j = 0; j < b_1.size(); ++j){
                    b_1(j) = this->block_vec(j,0);
                    b_2(j) = this->block_vec(j,1);
            }
            return 0; // for compiling, has no sense at all
        };

    // Iterative refinement with step limit
    double r_x,r_y,ce;
    if (store_info){
        info_steps.clear();
        info_single_step.clear();
        ellipse_parameters.clear();
        ellipse.get_radx(r_x);
        ellipse.get_rady(r_y);
        ellipse.get_center(ce);
        ellipse_parameters.push_back(r_x);
        ellipse_parameters.push_back(r_y);
        ellipse_parameters.push_back(ce);
        //
        info_single_step.push_back(ellipse_parameters);
        info_steps.push_back(info_single_step);
    }
    for (int steps = 0; steps < max_steps; ++steps) {
        bool found_new_ellipse = estimate_ellipse_chebyshev_arnoldi(cheb_n, block_matrix, detected_points, tol_fix, tol_variable, steps );
        if (store_info){
            info_steps.push_back(info_single_step);
        }
        if (!found_new_ellipse || steps == max_steps-1) {
            ellipse.create_opt_ellipse(detected_points);
            ellipse.get_radx(radius_tempx);ellipse.get_rady(radius_tempy);ellipse.get_center(center);ellipse.get_ecc(foci);
            if (steps == max_steps-1) {
                std::cout << "Warning: Maximum steps reached without convergence." << std::endl;
            } else {
                std::cout << "Optimal ellipse has been found! Converged in " << steps << " steps. " << std::endl;
            }
            std::cout << "Final ellipse:  radiusx = "<< radius_tempx << " radiusy = "<< radius_tempy << " center = "<< center << " foci = "<< foci << std::endl;
            std::cout << " " << std::endl;
            std::cout << "Completed calculating the optimal ellipse." << std::endl;
            std::cout << " " << std::endl;
            break;  // Exit loop if no new ellipse is found
        }
        if( detected_points.size() > points_size ){
        // start again with random vector
            krylov_vectors_u[0] = start_vec_krylov_u;
            krylov_vectors_v[0] = start_vec_krylov_v;
            get_start_vectors(krylov_vectors_u[0], krylov_vectors_v[0],true);
            points_size = detected_points.size();
        }
        else{
        // use approximate eigenvector for restart
            krylov_vectors_u[0] = start_vec_krylov_u;
            krylov_vectors_v[0] = start_vec_krylov_v;
            extract_real_part(krylov_vectors_u[0],krylov_vectors_v[0]);
        }
        //
        degree_max_reached = estimate_degree(degree,tol_fix);
        cheb_approx.set_chebyshev_tn_basis(degree);
        ellipse.get_radx(radius_tempx);ellipse.get_rady(radius_tempy); ellipse.get_center(center); ellipse.get_ecc(foci);
        std::cout << "Ellipse found in this step:  radiusx = "<< radius_tempx << " radiusy = "<< radius_tempy << " center = "<< center << " foci = "<< foci 
                  <<  " bd_p = " << bd_point << std::endl; 
    }
}

template class FindEllipse<1>;
template class FindEllipse<2>;

}
}