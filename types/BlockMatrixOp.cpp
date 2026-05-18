#include "BlockMatrixOp.h"

namespace Types {

template <int dim>
BlockMatrixOp<dim>::BlockMatrixOp(BlockMatrix<dim> &matrix)
    : block_matrix(matrix)
{
    size = matrix.get_size();
    a.reinit(size);
    b.reinit(size);
}

template <int dim>
int BlockMatrixOp<dim>::rows() const {
    return 2 * size;
}

template <int dim>
int BlockMatrixOp<dim>::cols() const {
    return 2 * size;
}

template <int dim>
void BlockMatrixOp<dim>::perform_op(const double *x_in, double *y_out) const {
    for (int i = 0; i < size; ++i) {
        a[i] = x_in[i];
        b[i] = x_in[i + size];
    }

    block_matrix.vmult(a, b, 1e-10);

    for (int i = 0; i < size; ++i) {
        y_out[i] = a[i];
        y_out[i + size] = b[i];
    }
}
template class BlockMatrixOp<1>;
template class BlockMatrixOp<2>;
} // namespace Types
