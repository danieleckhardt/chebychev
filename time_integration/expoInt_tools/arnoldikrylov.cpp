#include "arnoldikrylov.h"
namespace MavesS {
namespace TimeIntegration
{   
template<int dim, typename Number>
ArnoldiKrylov<dim,Number>::ArnoldiKrylov(const SparseMatrix<double>& A_, const SparseMatrix<double>& M_,  std::vector<Vector<Number>>& krylov_u, std::vector<Vector<Number>>& krylov_v, double& time_step_size_, int max_iterations_, bool weighted_sp)
    : 
    weighted_sp(weighted_sp)
    ,A(A_)
    ,M(M_)
    ,time_step_size(time_step_size_)
    ,max_iterations(max_iterations_)
    ,krylov_u(krylov_u)
    ,krylov_v(krylov_v)
    ,calc_eigen(eigenvalues, eigenvectors)
    {}

template<int dim, typename Number>
void ArnoldiKrylov<dim,Number>::get_matrix_funcxe1_krylov(std::vector<Number>& matrix_funcxe1_krylov_out) {
    matrix_funcxe1_krylov_out.resize(matrix_funcxe1_krylov.size());
    for(unsigned long i = 0; i <matrix_funcxe1_krylov.size() ; ++i ){
        matrix_funcxe1_krylov_out[i] = matrix_funcxe1_krylov[i];
        }
}

template<int dim, typename Number>
Number ArnoldiKrylov<dim,Number>::scalar_product(const Vector<Number>& a,
                                                 const Vector<Number>& b,
                                                 const SparseMatrix<double>& Matrix,
                                                 bool use_matrix ) 
{
    if(weighted_sp && use_matrix){
        Matrix.vmult(tmp_vec, a);
        mv_count++;
    }
    else{
        tmp_vec = a;
    }

    Number result = 0;
    for (unsigned int i = 0; i < b.size(); ++i)
    {
        if constexpr (std::is_same<Number, std::complex<double>>::value)
            result += std::conj(b[i]) * tmp_vec[i];
        else
            result += b[i] * tmp_vec[i];
    }

    return result;
}

//
//
// Only for debugging
//
//

template<int dim, typename Number>
void ArnoldiKrylov<dim, Number>::check_arnoldi_relation(BMatrixVariant MatrixMult, int iteration) const
    // Check the Arnoldi relation
    // Only for debugging 
{
    Assert(iteration <= max_iterations, ExcInternalError());

    // Step 1: Compute AV = [A * u_j, A * v_j] for j = 0..iteration-1
    std::vector<Vector<Number>> AV_u(iteration, Vector<Number>(A.m()));
    std::vector<Vector<Number>> AV_v(iteration, Vector<Number>(A.m()));
    
    for (int j = 0; j < iteration; ++j) {
        AV_u[j] = krylov_u[j];  // make a copy
        AV_v[j] = krylov_v[j];
        std::get<std::function<int(Vector<Number>&, Vector<Number>&)>>(MatrixMult)(AV_u[j], AV_v[j]);
    }

    // Step 2: Compute VH = V * H
    std::vector<Vector<Number>> VH_u(iteration, Vector<Number>(A.m()));
    std::vector<Vector<Number>> VH_v(iteration, Vector<Number>(A.m()));
    
    for (int j = 0; j < iteration; ++j) {
        VH_u[j] = 0.0;
        VH_v[j] = 0.0;
        for (int i = 0; i <= iteration; ++i) { // V_m+1
            VH_u[j].add(Hm(i, j), krylov_u[i]);
            VH_v[j].add(Hm(i, j), krylov_v[i]);
        }
    }

    // Step 3: Compute AV - VH and print residuals
    double max_norm = 0.0;
    for (int j = 0; j < iteration; ++j) {
        Vector<Number> res_u(AV_u[j]);
        Vector<Number> res_v(AV_v[j]);

        res_u.add(-1.0, VH_u[j]);
        res_v.add(-1.0, VH_v[j]);

        double norm = std::sqrt(res_u.l2_norm() * res_u.l2_norm() + res_v.l2_norm() * res_v.l2_norm());
        max_norm = std::max(max_norm, norm);
        
    }

    std::cout << "Max Arnoldi recurrence residual norm: " << max_norm ;
}

template<int dim, typename Number>
void ArnoldiKrylov<dim, Number>::calculate_projected_matrix(BMatrixVariant MatrixMult, int iteration)
{
    Assert(iteration <= max_iterations, ExcInternalError());
    projected_H.reinit(iteration, iteration);

    // Loop over columns of the projected matrix
    for (int j = 0; j < iteration; ++j)
    {
        // Apply the operator (A,M) to the current basis vectors
        Vector<Number> tmp_u = krylov_u[j];
        Vector<Number> tmp_v = krylov_v[j];
        std::get<std::function<int(Vector<Number>&, Vector<Number>&)>>(MatrixMult)(tmp_u, tmp_v);
        // Compute inner products <u_i, A*u_j> + <v_i, M*v_j>
        for (int i = 0; i < iteration; ++i)
        {
            Number hij = 0;
            const unsigned int n = krylov_u[i].size();

            if constexpr (std::is_same_v<Number, std::complex<double>>)
            {
                for (unsigned int l = 0; l < n; ++l)
                {
                    hij += std::conj(krylov_u[i][l]) * tmp_u[l];
                    hij += std::conj(krylov_v[i][l]) * tmp_v[l];
                }
            }
            else
            {
                for (unsigned int l = 0; l < n; ++l)
                {
                    hij += krylov_u[i][l] * tmp_u[l];
                    hij += krylov_v[i][l] * tmp_v[l];
                }
            }

            projected_H(i, j) = hij;
        }
    }
}

template<int dim, typename Number>
void ArnoldiKrylov<dim, Number>::check_projected_matrix(BMatrixVariant MatrixMult,FullMatrix<Number>& refH, int iteration) 
{
    Assert(iteration <= max_iterations, ExcInternalError());

    projected_H.reinit(iteration, iteration);

    for (int j = 0; j < iteration; ++j)
    {
        // Step 1: Compute A * (u_j, v_j)
        Vector<Number> tmp_u = krylov_u[j];
        Vector<Number> tmp_v = krylov_v[j];

        std::get<std::function<int(Vector<Number>&, Vector<Number>&)>>(MatrixMult)(tmp_u, tmp_v);

        for (int i = 0; i < iteration; ++i)
        {
            // Step 2: Compute <u_i, A * u_j> + <v_i, M * v_j>
            Number h_ij = scalar_product(tmp_u, krylov_u[i], A, false)
                        + scalar_product(tmp_v, krylov_v[i], M, false);

            projected_H(i, j) = h_ij;

            // Number h_ref = refH(i, j);
        }
    }

    // Optional: compute overall Frobenius norm of the error
    double frob_error = 0.0;
    for (int i = 0; i < iteration; ++i)
        for (int j = 0; j < iteration; ++j)
        {
            Number diff = projected_H(i, j) - refH(i, j);
            frob_error += std::abs(diff);  // works for both real and complex
        }

    frob_error = std::sqrt(frob_error);
    std::cout << "Frobenius norm of H_proj - Hm: " << frob_error ;
}

//
//
// 
//
//


template<int dim, typename Number>
void ArnoldiKrylov<dim,Number>::set_max_iterations(int max_iter)
{
    max_iterations = max_iter;
    krylov_u.resize(max_iterations, Vector<Number>(A.m()));
    krylov_v.resize(max_iterations, Vector<Number>(A.m()));
    tmp_vec.reinit(A.m());
    Hm.reinit(max_iterations + 1, max_iterations);
    store_Hm.reinit(max_iterations + 1);
    iterations_needed = max_iter;
}


template<int dim, typename Number>
void ArnoldiKrylov<dim,Number>::arnoldi_step(BMatrixVariant MatrixMult, int iteration)
{
    Vector<Number> um = krylov_u[iteration - 1];
    Vector<Number> vm = krylov_v[iteration - 1];

    
    mv_count += std::get<std::function<int(Vector<Number>&, Vector<Number>&)>>(MatrixMult)(um, vm);
    for (int i = 0; i < iteration; ++i) {
        store_Hm(i) = scalar_product(um, krylov_u[i], A)    
                + scalar_product(vm, krylov_v[i], M);
        um.add(-store_Hm(i), krylov_u[i]);
        vm.add(-store_Hm(i), krylov_v[i]);
    }
    
    store_Hm(iteration) = std::sqrt(scalar_product(um, um, A) + scalar_product(vm, vm, M));
    krylov_u[iteration] = um;
    krylov_v[iteration] = vm;
}

template<int dim, typename Number>
double ArnoldiKrylov<dim,Number>::calculate_residual(int iteration)
{
    return   std::fabs(Hm(iteration, iteration - 1)) * std::fabs(eigenvectors(iteration - 1,0));
}

template<int dim, typename Number>
double ArnoldiKrylov<dim,Number>::calculate_residual(std::function<std::complex<double>(std::complex<double>)> func, int iteration, bool rational)
{
    eigenvectors_invers.copy_from(eigenvectors);
    eigenvectors_invers.gauss_jordan(); // Replace with stable inversion if needed
    
    tmp_complex.resize(iteration);
    for (int i = 0; i < iteration; ++i){
        tmp_complex[i] = func(eigenvalues[i]) * eigenvectors_invers(i, 0);
    }
    
    matrix_funcxe1_krylov.reinit(iteration);
    for (int i = 0; i < iteration; ++i) {
        std::complex<double> tmp_entry = 0.0;
        for (int j = 0; j < iteration; ++j)
            tmp_entry += eigenvectors(i, j) * tmp_complex[j];
        matrix_funcxe1_krylov(i) = tmp_entry.real();
    }
    if(rational){
        Number res_rat_vec = 0.;
        for (int i = 0; i < iteration; ++i) {
            // std::cout << Hm_sub_matrix(iteration-1,i)*matrix_funcxe1_krylov(i) << std::endl;;
            res_rat_vec += Hm_sub_matrix(iteration-1,i)*matrix_funcxe1_krylov(i);
            
        }
        return norm_value_krylov/std::abs(rat_shift) *  std::abs(Hm(iteration, iteration - 1)) * std::abs(res_rat_vec);

    }
    else
        return norm_value_krylov *  std::abs(Hm(iteration, iteration - 1)) * std::abs(matrix_funcxe1_krylov(iteration - 1));
}

template<int dim, typename Number>
bool ArnoldiKrylov<dim, Number>::initialize_krylov(double tolerance)
{
    auto norm_squared = scalar_product(krylov_u[0], krylov_u[0], A) +
                        scalar_product(krylov_v[0], krylov_v[0], M);

    norm_value_krylov = std::sqrt(std::real(norm_squared));

    if (norm_value_krylov < tolerance * 1e-5)  // Possibly replace with absolute threshold
        return false;

    krylov_u[0] *= 1.0 / norm_value_krylov;
    krylov_v[0] *= 1.0 / norm_value_krylov;
    return true;
}

template<int dim, typename Number>
bool ArnoldiKrylov<dim, Number>::run_poly(
    BMatrixVariant MatrixMult,
    std::function<std::complex<double>(std::complex<double>)> func,
    double tolerance)
{
    mv_count = 0; 
    if (!initialize_krylov(tolerance))
        return true; // lucky breakdown

    bool converged = false;
    std::vector<int> subset = {0};

    for (int iteration = 1; iteration < max_iterations; ++iteration) {
        arnoldi_step(MatrixMult, iteration);

        for (int j = 0; j < iteration; ++j)
            Hm(j, iteration - 1) = store_Hm(j);
        Hm(iteration, iteration - 1) = store_Hm(iteration);

        krylov_u[iteration] *= 1. / Hm(iteration, iteration - 1);
        krylov_v[iteration] *= 1. / Hm(iteration, iteration - 1);

        // check_arnoldi_relation(MatrixMult, iteration); // only for debugging

        Hm_sub_matrix.reinit(iteration, iteration);
        Hm_sub_matrix.extract_submatrix_from(Hm, subset, subset);
        subset.push_back(iteration);

        int info = calc_eigen.calc_all_eig(Hm_sub_matrix, iteration);
        Assert(info >= 0, ExcInternalError());

        eps = func ? calculate_residual(func, iteration)
                       : calculate_residual(iteration);
        

        if(!use_stopping_criteria){
            eps = std::fabs(Hm(iteration, iteration -1));
        }

        if (eps < tolerance && iteration > 1) { 
            converged = true;
            iterations_needed = iteration;
            break;
        }
    }

    std::cout << "Arnoldi (poly) converged: " << converged
              << ", Iterations: " << iterations_needed
              << ", Max Iterations: " << max_iterations
              << ", Residual: " << eps << std::endl;
    return converged;
}

template<int dim, typename Number>
bool ArnoldiKrylov<dim, Number>::run_rational(
    BMatrixVariant MatrixMult,
    BMatrixVariant MatrixMult_inv,
    std::function<std::complex<double>(std::complex<double>)> func,
    double tolerance)
{
    mv_count = 0; 
    if (!initialize_krylov(tolerance))
        return true; // lucky breakdown

    bool converged = false;
    std::vector<int> subset = {0};
    double residual;
    for (int iteration = 1; iteration < max_iterations; ++iteration) {
        arnoldi_step(MatrixMult_inv, iteration);

        for (int j = 0; j < iteration; ++j)
            Hm(j, iteration - 1) = store_Hm(j);
        Hm(iteration, iteration - 1) = store_Hm(iteration);

        krylov_u[iteration] *= 1. / Hm(iteration, iteration - 1);
        krylov_v[iteration] *= 1. / Hm(iteration, iteration - 1);

        // check_arnoldi_relation(MatrixMult_inv, iteration); 
        
        Hm_sub_matrix.reinit(iteration, iteration);
        Hm_sub_matrix.extract_submatrix_from(Hm, subset, subset);
        subset.push_back(iteration);
        
        // Rational Krylov update
        Hm_sub_matrix.gauss_jordan(); // invert
        
        Vector<Number> tmp_u = krylov_u[iteration];
        Vector<Number> tmp_v = krylov_v[iteration];

        // for ( int i ; i < tmp_u.size() ; ++i  ){
        //     std::cout << tmp_u[i] << std::endl;
        // }
        // std::cout << std::endl;

        mv_count += std::get<std::function<int(Vector<Number>&, Vector<Number>&)>>(MatrixMult)(tmp_u, tmp_v);
        
        residual = std::sqrt(std::real( scalar_product(tmp_u, tmp_u, A, true) +
                                    scalar_product(tmp_v, tmp_v, M, true)) );
               
        // std::cout << residual<< std::endl;
        updated_matrix = Hm_sub_matrix;
        
        for (int i = 0; i < iteration ; ++i)
            updated_matrix(i, i) -= 1.0;
        
        updated_matrix *= -1/rat_shift;
        // std::cout << rat_shift << std::endl;
        
        // calculate_projected_matrix(MatrixMult,iteration);
        // Hm_sub_matrix =  projected_H;
        
        // check_projected_matrix(MatrixMult,Hm_sub_matrix, iteration); 
        // for (auto & entry : Hm_sub_matrix )
        //     std::cout << "new  " <<  entry.row() << " " << entry.column() << " " << entry.value() ; 
       
        // check_projected_matrix(MatrixMult,Hm_sub_matrix , iteration); 
        int info = calc_eigen.calc_all_eig(updated_matrix, iteration);
        Assert(info >= 0, ExcInternalError());

        
        eps = func ? calculate_residual(func, iteration,true)
                       : calculate_residual(iteration);
        

        if(!use_stopping_criteria){
            eps = std::fabs(Hm(iteration, iteration -1));
            residual = 1;
        }

        if (eps*residual < tolerance && iteration >1) { 
            converged = true;
            iterations_needed = iteration;
            break;
        }
    }

    std::cout << "Arnoldi (rational) converged: " << converged
              << ", Iterations: " << iterations_needed
              << ", Max Iterations: " << max_iterations
              << ", Residual: " << eps*residual << std::endl ;
    return converged;
}

template class ArnoldiKrylov<1, double>;
template class ArnoldiKrylov<2, double>;
template class ArnoldiKrylov<1, std::complex<double>>;
template class ArnoldiKrylov<2, std::complex<double>>;
}
}