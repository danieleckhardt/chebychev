/*
 * InverseMatrix.h
 *
 *  Created on: Jan 24, 2018
 *  
 *      Author: leibold, eckhardt
 */
 
#ifndef DATA_TYPES_INVERSEMATRIX_H_
#define DATA_TYPES_INVERSEMATRIX_H_
 
#include <deal.II/lac/sparse_decomposition.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_mic.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
 
#include "InverseMatrixBase.h"
#include "../data/Boundary.h"

#include <omp.h> // For parallelization
 
using namespace dealii;
using namespace MavesS;
using namespace Data;

namespace Types{
 
/**
 * Die Klasse Repräsentiert das Inverse einer Matrix. Bei jeder Multiplikation wird ein LGS mit der ursprünglichen Matrix gelöst.
 * Funktioniert nur spd Matrizen, da das CG Verfahren zum Lösen der Gleichungssysteme verwendet wird...
 */
template <int dim, typename number = double>
class InverseMatrix : public InverseMatrixBase<dim,number>{
public:
        InverseMatrix(SparseMatrix<number>& M,  DoFHandler<dim>&  dof_handler_macro, bool apply_bd = true, bool use_preconditioner = true, bool use_cg = true);
        InverseMatrix() = delete;
        // ~InverseMatrix() = default;
        // InverseMatrix(const InverseMatrix<dim>& M_inv ) = delete;
        // InverseMatrix& operator=(const InverseMatrix<dim>& M_inv ) = delete;
        // InverseMatrix(InverseMatrix<dim>&& M_inv ) = delete;
        // InverseMatrix& operator=(InverseMatrix<dim>&& M_inv ) = delete;

        unsigned int vmult(Vector<number> &dst,  Vector<number> &src, double tol_) override;
        unsigned int vmult_without_preconditioner(Vector<number> &dst,   const Vector<number> &src) ;
        unsigned int vmult_with_preconditioner(Vector<number> &dst,   const Vector<number> &src) ;
        unsigned int vmult_mass_lumping(Vector<number> &dst,   const Vector<number> &src) override;
        void create_mass_lumping_preconditioner();
        unsigned int vmult_mass_lumping_preconditioner(Vector<number> &dst,   const Vector<number> &src) ;
        unsigned int mmult(FullMatrix<number>& C, const FullMatrix<number>& B, double tol) override;
private:
        number get_row_sum(const SparseMatrix<number> &M, unsigned int row);
        void apply_boundary_conditions(Vector<number> &dst,  Vector<number> &src, SparseMatrix<number> &Matrix);
        unsigned int solve_system(Vector<number> &dst, const Vector<number> &src);
        unsigned int solve_system_with_preconditioner(Vector<number> &dst, const Vector<number> &src);
        SparseMatrix<number> &M;
        bool apply_bd;
        bool use_preconditioner;
        bool use_cg;
        DoFHandler<dim>&  dof_handler_macro;
        SparseILU<number> preconditioner;
        SparseILU<number> preconditioner_sym;
        int row_n;
        std::vector<number> row_sums{}; // Precomputed row sums for mass lumping
        // PreconditionSSOR<SparseMatrix<number>> preconditioner_sym ;
        SolverControl solver_control;
        SolverCG<Vector<number>> solver_cg;
        SolverGMRES<Vector<number>> solver_gmres;
        Vector<number> src_temp;
        Vector<number> lump_mass_vec; // size of system
        DiagonalMatrix<Vector<number>> lumped_mass;
        std::map<types::global_dof_index, number> boundary_values0;
        std::map<types::global_dof_index, number> boundary_values1;
};
 
} /* namespace Types */
 
# endif //DATA_TYPES_INVERSEMATRIX_H_