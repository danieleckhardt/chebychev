#include "calc_eigen.h"

namespace MavesS //Multi-Waves-Scale
{
namespace TimeIntegration
{   

template<int dim>
Calc_eigen<dim>::Calc_eigen( std::vector<std::complex<double>>& eigenvalues, FullMatrix<std::complex<double>>& eigenvectors)
:
    eigenvalues(eigenvalues)
    ,eigenvectors(eigenvectors)
{}

template<int dim>
int Calc_eigen<dim>::calc_all_eig(const FullMatrix<double>& A, int const size ){
    eigenvectors.reinit(size,size);
    eigenvalues.resize(size);
    std::vector<double> A_temp(size * size); 
    for ( int i = 0; i < size; ++i) {
        for ( int j = 0; j < size; ++j) {
            A_temp[i * size + j] = A(i, j);
        }
    }       
    int info ; 
    std::vector<double> wr(size), wi(size), vr(size*size); 

    info = LAPACKE_dgeev(LAPACK_ROW_MAJOR,'N','V',size,A_temp.data(),size, wr.data(),wi.data(),nullptr,size ,vr.data(),size);

    for ( int i = 0; i < size; ++i){
        eigenvalues[i] = std::complex<double>( wr[i], wi[i] );
    }
    int col = 0;
    // eigenvectors are stored column-wise, so eigenvectors(n,m) returns the n-th entry of the m-th eigenvector
    while (col < size) {
        if (wi[col] == 0.0) {  // Real eigenvalue
            for (int j = 0; j < size; ++j) {
                eigenvectors(j, col) = {vr[j * size + col], 0.0}; // dgeev saves eigenvectors row-wise
            }
            col++;
        } else {  // Complex pair (col and col+1)
            for (int j = 0; j < size; ++j) {
                // Eigenvector for wr[col] + i·wi[col]
                eigenvectors(j, col) = {
                    vr[j * size + col],
                    vr[j * size + (col + 1)]  // dgeev saves eigenvectors row-wise
                };
                // Eigenvector for wr[col] - i·wi[col] (conjugate)
                eigenvectors(j, col + 1) = {
                    vr[j * size + col ],
                    -vr[j * size + (col + 1)] // dgeev saves eigenvectors row-wise
                };
            }
            col += 2;  // Skip the next column (already handled)
        }
    }
    //Sort by absolute value of the eigenvalue (descending):
    std::vector<EigPair> eig_pairs(size);
    for (int i = 0; i < size; ++i) {
        eig_pairs[i].value = eigenvalues[i];
        eig_pairs[i].vector.resize(size);
        for (int j = 0; j < size; ++j) {
            eig_pairs[i].vector[j] = eigenvectors(j, i);
        }
    }

    std::sort(eig_pairs.begin(), eig_pairs.end(),
        [](const EigPair& a, const EigPair& b) {
            return std::abs(a.value) > std::abs(b.value);
        }
    );

    for (int i = 0; i < size; ++i) {
        eigenvalues[i] = eig_pairs[i].value;
        for (int j = 0; j < size; ++j) {
            eigenvectors(j, i) = eig_pairs[i].vector[j];
        }
    }

    return info;
}

template<int dim>
int Calc_eigen<dim>::calc_all_eig(const FullMatrix<std::complex<double>>& A, const int size) {
    eigenvectors.reinit(size, size);
    eigenvalues.resize(size);

    std::vector<std::complex<double>> A_temp(size * size);
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            A_temp[i * size + j] = A(i, j); // Row-major format

    std::vector<std::complex<double>> w(size);               // Eigenvalues
    std::vector<std::complex<double>> vl(size * size);       // Left eigenvectors (not used)
    std::vector<std::complex<double>> vr(size * size);       // Right eigenvectors

    // Compute eigenvalues and right eigenvectors only
    int info = LAPACKE_zgeev(LAPACK_ROW_MAJOR, 'N', 'V', size,
                             reinterpret_cast<lapack_complex_double*>(A_temp.data()), size,
                             reinterpret_cast<lapack_complex_double*>(w.data()),
                             nullptr, size,
                             reinterpret_cast<lapack_complex_double*>(vr.data()), size);

    if (info != 0) {
        std::cerr << "LAPACKE_zgeev failed with error code " << info << std::endl;
        return info;
    }

    // Copy eigenvalues
    for (int i = 0; i < size; ++i)
        eigenvalues[i] = w[i];

    // Copy eigenvectors into matrix form
    // Eigenvectors are stored column-wise, so eigenvectors(n,m) returns the n-th entry of the m-th eigenvector
    for (int col = 0; col < size; ++col)
        for (int row = 0; row < size; ++row)
            eigenvectors(row, col) = vr[row * size + col]; 

    // Sort eigenvalues/vectors by magnitude (descending)
    std::vector<EigPair> eig_pairs(size);
    for (int i = 0; i < size; ++i) {
        eig_pairs[i].value = eigenvalues[i];
        eig_pairs[i].vector.resize(size);
        for (int j = 0; j < size; ++j)
            eig_pairs[i].vector[j] = eigenvectors(j, i);
    }

    std::sort(eig_pairs.begin(), eig_pairs.end(),
        [](const EigPair& a, const EigPair& b) {
            return std::abs(a.value) > std::abs(b.value);
        });

    // Store sorted results
    for (int i = 0; i < size; ++i) {
        eigenvalues[i] = eig_pairs[i].value;
        for (int j = 0; j < size; ++j)
            eigenvectors(j, i) = eig_pairs[i].vector[j];
    }

    return 0; // success
}

template<int dim>
int Calc_eigen<dim>::calc_max_eig(Types::BlockMatrix<dim>& block_matrix, int const k, double const tol, std::string method) {
    if (method == "spectra") {
        return calc_max_eig_spectra(block_matrix, k, tol);
    } else if (method == "petsc") {
        return calc_max_eig_petsc(block_matrix, k, tol);
    } else {
        std::cerr << "Unknown method: " << method << std::endl;
        return -1; // Error
    }
}

template<int dim>
int Calc_eigen<dim>::calc_max_eig_spectra(Types::BlockMatrix<dim>& block_matrix, int const k, double const tol ){

    int info ; 
    int ncv = 3*k +1;
    int max_iter = 1000;
    Types::BlockMatrixOp<dim> op(block_matrix);
    Spectra::GenEigsSolver<Types::BlockMatrixOp<dim>> eigs(op, k, ncv);
    
    eigs.init();
    info = eigs.compute(Spectra::SortRule::LargestMagn, max_iter, tol);
    Eigen::VectorXcd eigvals = eigs.eigenvalues();  
    eigenvalues.resize(eigvals.size());
    for (int i = 0; i < eigvals.size(); ++i)
        eigenvalues[i] = eigvals[i];
    return info;
}

template<int dim>
int Calc_eigen<dim>::calc_max_eig_petsc(Types::BlockMatrix<dim>& block_matrix, int const k, double const tol)
{
    PetscErrorCode ierr;
    Mat A;
    EPS eps;
    PetscInt nconv;
    PetscScalar kr, ki;

    // Create shell matrix from block matrix
    ierr = create_shell_matrix<dim>(PETSC_COMM_WORLD, block_matrix, &A); CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Create eigensolver context
    ierr = EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = EPSSetOperators(eps, A, nullptr); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = EPSSetProblemType(eps, EPS_NHEP); CHKERRABORT(PETSC_COMM_WORLD, ierr); // Non-Hermitian

    // Set solver options
    ierr = EPSSetTolerances(eps, tol, 1000); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = EPSSetDimensions(eps, k, PETSC_DECIDE, PETSC_DECIDE); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = EPSSetFromOptions(eps); // allow command-line overrides

    // Solve the eigenproblem
    ierr = EPSSolve(eps); CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // Get number of converged eigenpairs
    ierr = EPSGetConverged(eps, &nconv); CHKERRABORT(PETSC_COMM_WORLD, ierr);

    eigenvalues.clear();
    for (PetscInt i = 0; i < std::min(k, nconv); ++i)
    {
        ierr = EPSGetEigenvalue(eps, i, &kr, &ki); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        eigenvalues.emplace_back(kr, ki);
    }

    // Cleanup
    ierr = EPSDestroy(&eps); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = MatDestroy(&A); CHKERRABORT(PETSC_COMM_WORLD, ierr);

    return (nconv >= k) ? 0 : 1; // Return 0 if successful
}

template class Calc_eigen<1>;
template class Calc_eigen<2>;

}
}

