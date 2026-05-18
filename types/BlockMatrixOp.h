#ifndef BLOCK_MATRIX_OP_H
#define BLOCK_MATRIX_OP_H

#include "BlockMatrix.h"
#include <deal.II/lac/vector.h>

namespace Types {
// This class is a wrapper around BlockMatrix to provide an interface compatible with Spectra's

template <int dim>
class BlockMatrixOp {
private:
    BlockMatrix<dim> &block_matrix;
    mutable dealii::Vector<double> a;
    mutable dealii::Vector<double> b;
    int size;

public:
    BlockMatrixOp(BlockMatrix<dim> &matrix);

    int rows() const;
    int cols() const;

    using Scalar = double;

    void perform_op(const double *x_in, double *y_out) const;
};

} // namespace Types

#endif // BLOCK_MATRIX_OP_H