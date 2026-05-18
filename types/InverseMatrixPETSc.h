#ifndef DATA_TYPES_INVERSEMATRIXPETSC_H_
#define DATA_TYPES_INVERSEMATRIXPETSC_H_

#include <deal.II/base/mpi.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <map>

#include "InverseMatrixBase.h"

namespace Types
{

using namespace dealii;

template <int dim, typename number = double>
class InverseMatrixPETSc :  public InverseMatrixBase<dim,number>
{
public:

    InverseMatrixPETSc(SparseMatrix<number> &M,
                       const DoFHandler<dim> &dof_handler,
                       bool apply_boundary_conditions = true,
                       bool use_cg = true,
                       bool use_pre = true );


    unsigned int vmult(Vector<number> &dst,
                        Vector<number> &src,
                        double tol_) override ;


    unsigned int vmult_mass_lumping(Vector<number> &dst,
                                     const Vector<number> &src) override ;

    unsigned int mmult(FullMatrix<number>& C,
                        const FullMatrix<number>& B, 
                        double tol) override ;                                

private:

    void apply_boundary_conditions(Vector<number> &dst,  Vector<number> &src, SparseMatrix<number> &Matrix);

    number get_row_sum(const SparseMatrix<number> &M,
                       unsigned int row) const;


private:

    SparseMatrix<number> &M;

    const DoFHandler<dim> &dof_handler;

    MPI_Comm mpi_communicator;

    unsigned int n_mpi_processes;

    unsigned int this_mpi_process;

    bool apply_bd;
    bool use_cg;
    bool use_pre;

    PETScWrappers::SparseMatrix petsc_matrix;
    SparsityPattern    sparsity_pattern;


    mutable PETScWrappers::MPI::Vector petsc_src;
    mutable PETScWrappers::MPI::Vector petsc_dst;

    PETScWrappers::PreconditionBoomerAMG petsc_amg;
    PETScWrappers::PreconditionICC petsc_icc;
    PETScWrappers::PreconditionILU petsc_ilu;
    PETScWrappers::PreconditionNone petsc_none;
    PETScWrappers::	PreconditionJacobi petsc_jac;

    mutable SolverControl solver_control;

    mutable PETScWrappers::SolverCG petsc_solver;
    mutable PETScWrappers::SolverGMRES petsc_solver_gmres;

    std::vector<number> row_sums;
    std::map<types::global_dof_index, number> boundary_values0;
    std::map<types::global_dof_index, number> boundary_values1;
};

}

#endif