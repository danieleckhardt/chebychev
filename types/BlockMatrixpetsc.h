#include <petscmat.h>
#include "BlockMatrix.h"

template<int dim>
struct BlockMatrixShellCtx {
    Types::BlockMatrix<dim>* block_matrix;
};

template<int dim>
PetscErrorCode BlockMatrixShellMult(Mat A, Vec x, Vec y)
{
    PetscErrorCode ierr;
    void* ctx = nullptr;
    ierr = MatShellGetContext(A, &ctx); CHKERRQ(ierr);

    auto* shell_ctx = static_cast<BlockMatrixShellCtx<dim>*>(ctx);
    auto& block_matrix = *(shell_ctx->block_matrix);

    // Get vector size
    PetscInt n;
    ierr = VecGetSize(x, &n); CHKERRQ(ierr);

    if (n % 2 != 0)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Input vector size must be even (a/b components)");

    int half_size = n / 2;
    // Access raw data
    const PetscScalar* x_array;
    PetscScalar* y_array;
    ierr = VecGetArrayRead(x, &x_array); CHKERRQ(ierr);
    ierr = VecGetArray(y, &y_array); CHKERRQ(ierr);

    // Wrap input vectors
    Vector<double> A_dealii(half_size);
    Vector<double> B_dealii(half_size);
    for (int i = 0; i < half_size; ++i) {
        A_dealii[i] = x_array[i];
        B_dealii[i] = x_array[half_size + i];
    }

    // Output vectors
    // Vector<double> A_out(half_size);
    // Vector<double> B_out(half_size);


    block_matrix.vmult( A_dealii, B_dealii, 1e-10);

    for (int i = 0; i < half_size; ++i) {
        y_array[i] = A_dealii[i];
        y_array[half_size + i] = B_dealii[i];
    }

    ierr = VecRestoreArrayRead(x, &x_array); CHKERRQ(ierr);
    ierr = VecRestoreArray(y, &y_array); CHKERRQ(ierr);

    return 0;
}

template<int dim>
PetscErrorCode BlockMatrixShellDestroy(Mat A)
{
    void* ctx = nullptr;
    PetscErrorCode ierr = MatShellGetContext(A, &ctx); CHKERRQ(ierr);

    auto* shell_ctx = static_cast<BlockMatrixShellCtx<dim>*>(ctx);
    delete shell_ctx;

    return 0;
}

template<int dim>
PetscErrorCode create_shell_matrix(MPI_Comm comm, Types::BlockMatrix<dim>& block_matrix, Mat* A_out)
{
    PetscErrorCode ierr;
    PetscInt n = block_matrix.get_size();
    PetscInt mat_size = 2 * n;

    auto* shell_ctx = new BlockMatrixShellCtx<dim>{ &block_matrix };

    Mat A;
    ierr = MatCreateShell(comm, mat_size, mat_size, mat_size, mat_size, shell_ctx, &A); CHKERRQ(ierr);

    ierr = MatShellSetOperation(A, MATOP_MULT, (void(*)(void))BlockMatrixShellMult<dim>); CHKERRQ(ierr);
    ierr = MatShellSetOperation(A, MATOP_DESTROY, (void(*)(void))BlockMatrixShellDestroy<dim>); CHKERRQ(ierr);

    *A_out = A;
    return 0;
}
