#ifndef ARNOLDIKRYLOV_H_
#define ARNOLDIKRYLOV_H_

#include <cmath>
#include <iomanip>
#include <functional> // Required for std::function
#include <chrono>


#include "../../types/InverseMatrix.h"
#include "../../types/BlockMatrix.h"
#include "../../types/BlockMatrixOp.h"
#include "calc_eigen.h"

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <lapacke.h> // LAPACK


using namespace dealii;
using namespace std::complex_literals; // needs c++14
using BMatrixVariant = std::variant<
    std::function<int(FullMatrix<std::complex<double>>&)>,
    std::function<int(Vector<double>&, Vector<double>&)>,
    std::function<int(Vector<std::complex<double>>&, Vector<std::complex<double>>&)>
>;

namespace MavesS //Multi-Waves-Scale
{
namespace TimeIntegration
{

template<int dim, typename Number = double>
class ArnoldiKrylov {
public:
    ArnoldiKrylov(const SparseMatrix<double>& A, const SparseMatrix<double>& M, std::vector<Vector<Number>>& krylov_u, 
                    std::vector<Vector<Number>>& krylov_v, double& time_step_size, int max_iterations = 10, bool weighted_sp = true);

    void set_max_iterations(int max_iter);
    //
    bool run_poly(
        BMatrixVariant MatrixMult,
        std::function<std::complex<double>(std::complex<double>)> func,
        double tolerance);

    bool run_rational(
        BMatrixVariant MatrixMult,
        BMatrixVariant MatrixMult_inv,
        std::function<std::complex<double>(std::complex<double>)> func,
        double tolerance);
    //
    Number get_residual() const { return eps; }
    int get_iterations() const { return iterations_needed; }
    double get_norm () const { return norm_value_krylov; }
    // void get_Hm(FullMatrix<Number>& Hm_out) const {  Hm_out = Hm_sub_matrix;}
    std::complex<double> get_max_eigenvalue() const { if (eigenvalues.empty()) {  return std::complex<double>(0.0, 0.0);} return eigenvalues.front();   }
    void get_matrix_funcxe1_krylov(std::vector<Number>& matrix_funcxe1_krylov_out);
    void set_time_step_size(double& time_step_size_) { time_step_size = time_step_size_; }
    void get_count_mv(int& mv_count_out) const { mv_count_out = mv_count; }
    void set_rat_shift(double shift) { rat_shift = shift; }
    void set_weighted_sp(bool weighted) { weighted_sp = weighted; }
private:
    bool weighted_sp;
    bool use_stopping_criteria = true;
    int mv_count = 0;
    double rat_shift = 1.;
    bool initialize_krylov(double tolerance);
    void arnoldi_step(BMatrixVariant MatrixMult, int iteration );
    double calculate_residual(int iteration);
    double calculate_residual(std::function<std::complex<double>(std::complex<double>)> func, int iteration, bool rational = false);
    Number scalar_product(const Vector<Number>& a, const Vector<Number>& b, const SparseMatrix<double>& Matrix, bool use_matrix = true) ;
    void check_arnoldi_relation(BMatrixVariant MatrixMult, int iteration) const;
    void check_projected_matrix(BMatrixVariant MatrixMult,FullMatrix<Number>& refH, int iteration) ;
    void calculate_projected_matrix(BMatrixVariant MatrixMult,  int iteration) ;

    // Matrices and vectors
    const SparseMatrix<double>& A;
    const SparseMatrix<double>& M;
    double& time_step_size;
    double norm_value_krylov = 1.0;
    
    int max_iterations;
    int iterations_needed = 0;
    double eps = 0.0;
    
    std::vector<Vector<Number>>& krylov_u;
    std::vector<Vector<Number>>& krylov_v;
    Vector<Number> matrix_funcxe1_krylov;
    //
    Vector<Number> store_Hm, tmp_vec;
    FullMatrix<Number> Hm, Hm_sub_matrix;
    FullMatrix<Number> updated_matrix;
    FullMatrix<Number> projected_H; // for debugging

    FullMatrix<std::complex<double>> eigenvectors;
    FullMatrix<std::complex<double>> eigenvectors_invers;
    std::vector<std::complex<double>> eigenvalues, tmp_complex;

    Calc_eigen<dim> calc_eigen;
};

}   
}
#endif /* ARNOLDIKRYLOV_H_ */
