#include "InverseMatrixPETSc.h"

namespace Types
{

using namespace dealii;

template <int dim, typename number>
InverseMatrixPETSc<dim,number>::InverseMatrixPETSc(
        SparseMatrix<number> &M_,
        const DoFHandler<dim> &dof_handler_,
        bool apply_boundary_conditions_,
        bool use_cg_,
        bool use_pre )
:
M(M_),
dof_handler(dof_handler_),
apply_bd(apply_boundary_conditions_),
use_cg(use_cg_),
use_pre(use_pre),
solver_control(1000,1e-10),
petsc_solver(solver_control),
petsc_solver_gmres(solver_control),
row_sums(M.m())
{
    mpi_communicator = MPI_COMM_WORLD;

    const unsigned int n = M.m();

    /*
     * Build PETSc matrix
     */
    petsc_matrix.reinit(M.get_sparsity_pattern());

    for (unsigned int i=0;i<n;++i)
        for (auto it = M.begin(i); it != M.end(i); ++it)
            petsc_matrix.set(i,it->column(),it->value());

    petsc_matrix.compress(VectorOperation::insert);

    /*
     * PETSc vectors
     */
    petsc_src.reinit(mpi_communicator,n,n);
    petsc_dst.reinit(mpi_communicator,n,n);
    

    /*
     * AMG Preconditioner
     */
    PETScWrappers::PreconditionBoomerAMG::AdditionalData amg_data;
    PETScWrappers::PreconditionICC::AdditionalData icc_data;
    PETScWrappers::PreconditionILU::AdditionalData ilu_data;
    PETScWrappers::PreconditionNone::AdditionalData none_data;
    if(use_cg){
        amg_data.symmetric_operator = true;
    }
    

    petsc_amg.initialize(petsc_matrix,amg_data);
    petsc_icc.initialize(petsc_matrix,icc_data);
    petsc_ilu.initialize(petsc_matrix,ilu_data);
    petsc_none.initialize(petsc_matrix,none_data);

    /*
     * Precompute row sums for mass lumping
     */
    for (unsigned int i=0;i<n;++i)
        row_sums[i] = get_row_sum(M,i);
}



template <int dim, typename number>
number InverseMatrixPETSc<dim,number>::get_row_sum(
        const SparseMatrix<number> &matrix,
        unsigned int row) const
{
    number sum = 0;

    for (auto it = matrix.begin(row); it != matrix.end(row); ++it)
        sum += it->value();

    return sum;
}



template <int dim, typename number>
void InverseMatrixPETSc<dim,number>::apply_boundary_conditions(
        Vector<number> &dst,
        Vector<number> &src,
        SparseMatrix<number> &matrix)
{
    if (boundary_values0.empty())
        VectorTools::interpolate_boundary_values(
            dof_handler,
            0,
            Functions::ConstantFunction<dim,number>(0.0),
            boundary_values0);

    MatrixTools::apply_boundary_values(boundary_values0,matrix,dst,src);

    if constexpr (dim == 1)
    {
        if (boundary_values1.empty())
            VectorTools::interpolate_boundary_values(
                dof_handler,
                1,
                Functions::ConstantFunction<dim,number>(0.0),
                boundary_values1);

        MatrixTools::apply_boundary_values(boundary_values1,matrix,dst,src);
    }
}



template <int dim, typename number>
unsigned int InverseMatrixPETSc<dim,number>::vmult(
        Vector<number> &dst,
        Vector<number> &src,
        double tol)
{
    solver_control.set_tolerance(tol);
    if(apply_bd)
        apply_boundary_conditions(dst, src, M);

    /*
     * Copy RHS to PETSc vector
     */
    petsc_src = src;
    petsc_src.compress(VectorOperation::insert);

    /*
     * Solve system
     */
    if(use_cg){
        if(use_pre){
            petsc_solver.solve(
                petsc_matrix,
                petsc_dst,
                petsc_src,
                petsc_icc);
        }
        else{
            petsc_solver.solve(
                petsc_matrix,
                petsc_dst,
                petsc_src,
                petsc_none);
        }
    }
    else{
        if(use_pre){
            petsc_solver_gmres.solve(
                petsc_matrix,
                petsc_dst,
                petsc_src,
                petsc_ilu);
        }
        else{
            petsc_solver_gmres.solve(
                petsc_matrix,
                petsc_dst,
                petsc_src,
                petsc_none);
        }
    }

    /*
     * Copy result back
     */
    dst = petsc_dst;

    return solver_control.last_step();
}



template <int dim, typename number>
unsigned int InverseMatrixPETSc<dim,number>::vmult_mass_lumping(
        Vector<number> &dst,
        const Vector<number> &src)
{
    const unsigned int n = src.size();

    dst.reinit(n);

    for (unsigned int i=0;i<n;++i)
    {
        if (row_sums[i] != 0.0)
            dst[i] = src[i] / row_sums[i];
        else
            dst[i] = 0.0;
    }

    return 1;
}



template <int dim, typename number>
unsigned int InverseMatrixPETSc<dim,number>::mmult(
        FullMatrix<number> &C,
        const FullMatrix<number> &B,
        double tol)
{
    const unsigned int n = B.n();

    Vector<number> rhs(n);
    Vector<number> sol(n);

    for (unsigned int i=0;i<n;++i)
    {
        for (unsigned int j=0;j<n;++j)
            rhs[j] = B(j,i);

        unsigned int info = vmult(sol,rhs,tol);

        if (info == 0)
            return 0;

        for (unsigned int j=0;j<n;++j)
            C(j,i) = sol[j];
    }

    return 1;
}



/*
 * Explicit template instantiations
 */

template class InverseMatrixPETSc<1,double>;
template class InverseMatrixPETSc<2,double>;

}