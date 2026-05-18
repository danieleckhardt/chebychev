#ifndef DATA_TYPES_BLOCKMATRIX_H_
#define DATA_TYPES_BLOCKMATRIX_H_
 
#include <deal.II/lac/sparse_decomposition.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_mic.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include "InverseMatrix.h"
#include "InverseMatrixPETSc.h"
  
using namespace dealii;
using namespace MavesS;

namespace Types{
template <int dim>
class BlockMatrix {
    // This class performs the Block Matrix Vector multipliction
    public:
        BlockMatrix(SparseMatrix<double>& A,SparseMatrix<double>& B, SparseMatrix<double>& M,  DoFHandler<dim>&  dof_handler, const double factor, bool apply_bd_conditions=true, bool sym_damp = false, bool use_pre_for_mass = true);
        BlockMatrix() = delete;
        int get_size();
        void get_matrices(const SparseMatrix<double>*& A_out, const SparseMatrix<double>*& B_out, const SparseMatrix<double>*& M_out, const InverseMatrix<dim, double>*& M_inv_out);
        int vmult(Vector<double> &a,  Vector<double> &b, double tol);
        int vmult_shift(Vector<double> &a,  Vector<double> &b, double tol);
        int vmult(Vector<double> &a,  Vector<double> &b, const Vector<double> &a_old,  const Vector<double> &b_old, double tol);
        int vmult_weak_op(Vector<double> &a,  Vector<double> &b);
        int vmult_weak_op(Vector<double> &a,  Vector<double> &b, const Vector<double> &a_old,  const Vector<double> &b_old);
        int vmult_rat(Vector<double> &a,  Vector<double> &b, double tol);
        int vmult_rat(Vector<double> &a,  Vector<double> &b, const Vector<double> &a_old,  const Vector<double> &b_old, double tol);
        void vmultT(Vector<double> &a,  Vector<double> &b,const Vector<double> &a_old,  const Vector<double> &b_old, double tol);
        int vmult_complex_split(Vector<std::complex<double>> &a,  Vector<std::complex<double>> &b, double tol, bool rational = false);
        int vmult_complex_split(Vector<std::complex<double>> &a,  Vector<std::complex<double>> &b, const Vector<std::complex<double>> &a_old,  const Vector<std::complex<double>> &b_old, double tol);
        int vmult_complex_split_weak_op(Vector<std::complex<double>> &a,  Vector<std::complex<double>> &b, const Vector<std::complex<double>> &a_old,  const Vector<std::complex<double>> &b_old);
        int vmultBlock(FullMatrix<std::complex<double>>& block_vec, double tol );
        void set_factor(double factor);
        void set_rat_shift(double shift_in){this->rat_shift = shift_in;}
        void get_rat_shift(double& shift_out){shift_out = rat_shift;};
    private:
        void update_rat_krylov_matrix();
        SparseMatrix<double>& A;
        SparseMatrix<double>& B;
        SparseMatrix<double>& M;
        SparseMatrix<double> rat_krylov;
        std::unique_ptr<InverseMatrixPETSc<dim, double>> rat_krylov_inv;
        bool apply_bd_conditions;
        bool sym_damp;
        double factor;
        double rat_shift = 1.;
        double cost_pre = 1;
        Vector<double> old_a;
        Vector<double> old_b;
        Vector<double> a_real, a_imag;
        Vector<double> b_real, b_imag;
        DoFHandler<dim>&  dof_handler;
        InverseMatrix<dim,double> M_inv_dealii;
        InverseMatrixPETSc<dim,double> M_inv;
        int size;

};
}
# endif //DATA_TYPES_BLOCKMATRIX_H_