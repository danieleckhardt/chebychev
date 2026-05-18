/*
 * InverseMatrix.cpp
 *
 *  Created on: Jan 24, 2018
 *      Author: leibold, eckhardt
 */
 
#include "InverseMatrix.h"
 
namespace Types{
 
template <int dim, typename number>
InverseMatrix<dim, number>::InverseMatrix(SparseMatrix<number>& M,
                                          DoFHandler<dim>& dof_handler_macro, bool apply_bd, bool use_preconditioner, bool use_cg)
    : M(M),
      apply_bd(apply_bd),
      use_preconditioner(use_preconditioner),
      use_cg(use_cg),
      dof_handler_macro(dof_handler_macro),
      preconditioner(),
      preconditioner_sym(),
      row_n(M.m()),
      row_sums(M.m()),
      solver_control(1000, 1e-8), 
      solver_cg(solver_control), 
      solver_gmres(solver_control), 
      src_temp(M.m()),
      lump_mass_vec(M.m())
    {
        if(use_cg)
            preconditioner_sym.initialize(M);
        else
            preconditioner.initialize(M);
           
        // Precompute row sums for mass lumping 
        row_sums[0] = M(0, 0); // Initialize first and last row sum (Dirichlet boundary condition)
        row_sums[row_n-1] = M(row_n-1, row_n-1); // Last row sum
        
        for (int i = 1; i < row_n-1; ++i){
            row_sums[i] = get_row_sum(M, i);
        }
        
        for (int i = 0; i < row_n; ++i) {
            lump_mass_vec[i] = 1. / row_sums[i]; // Store row sums in mass diagonal vector
            // std::cout << "lump_mass_vec " << i << " : " << row_sums[i] ;
        }
        lumped_mass.reinit(lump_mass_vec);
    }

 
template <int dim, typename number>
void InverseMatrix<dim, number>::apply_boundary_conditions(Vector<number>& dst,
                                                           Vector<number>& src,
                                                           SparseMatrix<number>& Matrix) {
    // Cache boundary conditions if not already done
    if (boundary_values0.empty()) {
        VectorTools::interpolate_boundary_values(dof_handler_macro, 0,
                                                 Functions::ConstantFunction<dim,number>(0.0), boundary_values0);
    }
    MatrixTools::apply_boundary_values(boundary_values0, Matrix, dst, src);

    if constexpr (dim == 1) {
        if (boundary_values1.empty()) {
            VectorTools::interpolate_boundary_values(dof_handler_macro, 1,
                                                      Functions::ConstantFunction<dim,number>(0.0), boundary_values1);
        }
        MatrixTools::apply_boundary_values(boundary_values1, Matrix, dst, src);
    }
}

template <int dim,typename number>
number InverseMatrix<dim, number>::get_row_sum(const SparseMatrix<number>& M, unsigned int row)
{
    number sum = 0;
    for (auto it = M.begin(row); it != M.end(row); ++it)
        sum += std::abs(it->value());
    return sum;
}


template <int dim, typename number>
unsigned int InverseMatrix<dim, number>::vmult(Vector<number> &dst,  Vector<number> &src, double tol_) {
    solver_control.set_tolerance(tol_);
    if(apply_bd)
        apply_boundary_conditions(dst, src, M); // Apply boundary conditions to dst and src
    if(use_preconditioner)
        return vmult_with_preconditioner(dst, src);
    else
        return vmult_without_preconditioner(dst, src); // best practice 
}

template <int dim, typename number>
unsigned int InverseMatrix<dim, number>::vmult_without_preconditioner(Vector<number>& dst, const Vector<number>& src) {
    return solve_system(dst, src);
}

template <int dim, typename number>
unsigned int InverseMatrix<dim, number>::vmult_with_preconditioner(Vector<number> &dst, const Vector<number> &src) {
    return solve_system_with_preconditioner(dst, src);
}

template <int dim, typename number>
unsigned int InverseMatrix<dim, number>::vmult_mass_lumping(Vector<number>& dst, const Vector<number>& src) {

    const unsigned int n_rows = M.m();
    
    bool failure = false;
    
    for (unsigned int i = 1; i < n_rows - 1; ++i) {
        const number sum_row = row_sums[i];
        if (sum_row != 0.0)
            dst(i) = src(i) / sum_row;
        else {
            std::cerr << "Zero row sum at row " << i << "!" << std::endl;
            failure = true;
            dst(i) = 0.0;
        }
    }
    dst(0) = 0; // preserve BC value if externally set
    dst(n_rows - 1) = 0; // preserve BC value

    return failure ? 0 : 1;
}

template <int dim, typename number>
unsigned int InverseMatrix<dim, number>::vmult_mass_lumping_preconditioner(Vector<number>& dst, const Vector<number>& src) {
    if(use_cg){
        solver_cg.solve(M, dst, src, lumped_mass);
    }
    else{
        solver_gmres.solve(M, dst, src, lumped_mass);
    }
    return solver_control.last_step();
}

template <int dim, typename number>
unsigned int InverseMatrix<dim, number>::solve_system(Vector<number>& dst, const Vector<number>& src) {
    try {
        if(use_cg){
            solver_cg.solve(M, dst, src, PreconditionIdentity());
        }
        else{
            solver_gmres.solve(M, dst, src, PreconditionIdentity());
        }
    } catch (std::exception &e) {
        std::cerr << "Solver failed: " << e.what() << std::endl;
        return 0; // Indicate failure
    }

    return solver_control.last_step();
}
template <int dim, typename number>
unsigned int InverseMatrix<dim, number>::solve_system_with_preconditioner(Vector<number>& dst, const Vector<number>& src) {
    try {
 
        if(use_cg){
            solver_cg.solve(M, dst, src, preconditioner_sym);
        }
        else{
            solver_gmres.solve(M, dst, src, preconditioner);
        }

    } catch (std::exception &e) {
        std::cerr << "Solver failed: " << e.what() << std::endl;
        return 0; // Indicate failure
    }

    return solver_control.last_step();
}

template <int dim, typename number>
unsigned int InverseMatrix<dim,number>::mmult(FullMatrix<number>& C,
                                                const FullMatrix<number>& B, double tol){
    const unsigned int size = C.n();
    Vector<number> tmp(size);

    for (unsigned int i = 0; i < size; ++i) {
        for (unsigned int j = 0; j < size; ++j)
            tmp(j) = B(j, i);

        unsigned int info = vmult(tmp, tmp, tol);
        if (info == 0)
            return 0;

        for (unsigned int j = 0; j < size; ++j)
            C(j, i) = tmp(j);
    }

    return 1;
}
 
template class InverseMatrix<1, double>;
template class InverseMatrix<2, double>;
// template class InverseMatrix<1, std::complex<double>>;
// template class InverseMatrix<2, std::complex<double>>;
} /* namespace Types */
