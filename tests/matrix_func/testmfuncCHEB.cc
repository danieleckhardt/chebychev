#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/matrix_tools.h>

#include "../../data/ExactSolution.h"
#include "../../data/InitialConditions.h"
#include "../../types/InverseMatrix.h"
#include "../../types/Ellipse.h"

#include "../../time_integration/expoInt_tools/expoInt.h"
#include "../../time_integration/expoInt_tools/chebyshev_approx.h"
#include "../../tools/Output.h"

#include <iostream>
#include <random>
#include <lapacke.h>
#include <complex>
#include <functional>


using namespace dealii;
using namespace MavesS;
using namespace Data;
using namespace TimeIntegration;

template< typename VectorType>
void generateRandomVector(VectorType& vec, double min_val, double max_val) {
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<> dis(min_val, max_val);

    for (unsigned int i = 0; i < vec.size(); ++i) {
        vec[i] = dis(gen);  // Assign random value
    }
}

template <typename VectorType>
void generateDiscreteSine(VectorType &vec, double frequency = 1.0)
{
    const unsigned int N = vec.size();
    for (unsigned int i = 0; i < N; ++i)
    {
        double x = static_cast<double>(i ) / (N-1 ); // normalized grid point in (0, 1)
        vec[i] = std::sin(frequency * numbers::PI * x*x);
    }
}


std::complex<double> phi1(std::complex<double> x)
{
    // //Numerator and denominator coefficients of the (6,6)-Padé approximation to Phi1
    std::vector<double> Nkoeff;
    std::vector<double> Dkoeff;
    //Coefficients of the Pade Approximation
    Nkoeff.resize(7);
    Nkoeff[0] = 1.;
    Nkoeff[1] = 1./26;
    Nkoeff[2] = 5./156;
    Nkoeff[3] = 1./858;
    Nkoeff[4] = 1./5720;
    Nkoeff[5] = 1./205920;
    Nkoeff[6] = 1./8648640;
    Dkoeff.resize(7);
    Dkoeff[0] = 1.;
    Dkoeff[1] = -6./13;
    Dkoeff[2] = 5./52;
    Dkoeff[3] = -5./429;
    Dkoeff[4] = 1./1144;
    Dkoeff[5] = -1./25740;
    Dkoeff[6] = 1./1235520;

    if(std::abs(x)>1e-4)
        return  (exp(x)-1.)/x;
    else if(std::abs(x) > 1e-16){
            std::complex<double> X   = x;
            std::complex<double> Phi = Nkoeff[0] + x * Nkoeff[1];
            std::complex<double> D   = Dkoeff[0] + x * Dkoeff[1];
            for(int k = 2; k<=6; k++)
            {
                    X   = x * X;
                    Phi = Phi + Nkoeff[k] * X;
                    D   = D   + Dkoeff[k] * X;
            }
            return Phi / D;
    }
    else
    return 1.;

}

template <int dim>
class testChebprotype
{
    public:
        testChebprotype();
        double run();
        double tol;
    private:
        int matrix_vector_product(Vector<double>& a, Vector<double>& b );
        int matrix_vector_product_complex_split(Vector<std::complex<double>>& a, Vector<std::complex<double>>& b );
        int matrix_matrix_product(FullMatrix<std::complex<double>>& A ) ;
        void setup_system();
        void create_ref_matrix_func(FullMatrix<double>& A, Vector<double>& b, Vector<double>& ref_vec_out);
        double error_calc(Vector<double>& ref_vec_long, Vector<double>& calc_vec1, Vector<double>& calc_vec2);

        Triangulation<dim> triangulation;

        DoFHandler<dim>    dof_handler;
        FE_Q<dim>          fe; 
        SparsityPattern    sparsity_pattern;

        SparseMatrix<double>             laplace_matrix;
        SparseMatrix<double>             homogeneous_matrix;
        SparseMatrix<double>             damping_matrix;
        SparseMatrix<double>             mass_matrix;
        FullMatrix<std::complex<double>> eigenvectors_ref;
        FullMatrix<std::complex<double>> eigenvectors_inverse_ref;
        FullMatrix<double>               ref_matrix_func;
        FullMatrix<double>               identity_m;
        
        AffineConstraints<double> constraints;
        // random vectors
        Vector<double> b_vec1; 
        Vector<double> b_vec2;
        // [b_vec1, b_vec2]
        Vector<double>   ref_vec_b;
        Vector<double>   ref_vec_out;
        // setup blockmatrix
        std::vector<int> first_index_set;
        std::vector<int> second_index_set;
        //
        std::vector<std::complex<double>> eigenvalues;

        bool  use_random_vector = false;
        double time_step_size;
        int    size;
        int    degree;

        ExpoInt<dim> expoint;
        Types::Ellipse<dim> ellipse;
        Output<dim,FullMatrix<double>> output_class;
        Output<dim,FullMatrix<std::complex<double>>> output_class_complex;

};

template<int dim>
testChebprotype<dim>::testChebprotype()
:   
    dof_handler(triangulation)
   ,fe(2)
   ,expoint(fe, dof_handler,constraints,homogeneous_matrix,damping_matrix,mass_matrix,time_step_size,0)

{}

template<int dim> 
int testChebprotype<dim>::matrix_vector_product(Vector<double>& a, Vector<double>& b ) {
    // factor *  [ [0 , I] [ -M^-1*homogeneous_matrix  , -M^-1* damping_matrix]  ] * [a , b]
    Vector<double> old_a(a);
    Vector<double> old_b(b);
    Types::InverseMatrix<dim> M_inv(mass_matrix,  dof_handler,false);

    //
    homogeneous_matrix.vmult(a,old_a);
    damping_matrix.vmult(b,old_b);
    b += a;
    b *= -time_step_size;
    M_inv.vmult(b, b, 1e-10);  
    // 
    a = old_b; 
    a *= time_step_size; 

    return 0;
}

template<int dim> 
int testChebprotype<dim>::matrix_vector_product_complex_split(Vector<std::complex<double>>& a, Vector<std::complex<double>>& b ) {
    // [ [0 , I] [ -M^-1*homogeneous_matrix  , -M^-1* damping_matrix]  ] * [a , b]
    const unsigned int size = a.size();

    Vector<double> a_real(size), a_imag(size);
    Vector<double> b_real(size), b_imag(size);

    for (unsigned int i = 0; i < size; ++i)
    {
        a_real[i] = a[i].real();
        a_imag[i] = a[i].imag();
        b_real[i] = b[i].real();
        b_imag[i] = b[i].imag();
    }

    matrix_vector_product(a_real,b_real);
    matrix_vector_product(a_imag,b_imag);

    for (unsigned int i = 0; i < size; ++i)
    {
        a[i] = std::complex<double>(a_real[i],a_imag[i]);
        b[i] = std::complex<double>(b_real[i],b_imag[i]);
    }

    return 0;

}

template<int dim> 
int testChebprotype<dim>::matrix_matrix_product(FullMatrix<std::complex<double>>& A ) {
    // [ [0 , I] [ -M^-1*homogeneous_matrix  , -M^-1* damping_matrix]  ] * [a , b]
    Vector<std::complex<double>> a(A.m());
    Vector<std::complex<double>> b(A.m());
    for(int i = 0; i < A.m(); ++i){
        a(i) = A(i,0);
        b(i) = A(i,1);
    }
    matrix_vector_product_complex_split(a,b); //faster 
    for(int i = 0; i < A.m(); ++i){
        A(i,0) = a(i);
        A(i,1) = b(i);
    }

    return 0;
}

template<int dim>
void testChebprotype<dim>::setup_system() {
    
    double coef_a = 0.5; // != 0
    double coef_damping = 0.3;
    //
    dof_handler.distribute_dofs(fe);
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();
    size = dof_handler.n_dofs();
    DynamicSparsityPattern dsp(size);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
    sparsity_pattern.copy_from(dsp);
    
    b_vec1.reinit(size);
    b_vec2.reinit(size);
    if (use_random_vector){
        generateRandomVector(b_vec1,0.,1.);
        generateRandomVector(b_vec2,0.,1.);
    }
    else{
        generateDiscreteSine(b_vec1,1.0);
        generateDiscreteSine(b_vec2,1.0);
        b_vec2 *= 3.;
    }

    ref_vec_b.reinit(2*size);
    for ( int i = 0; i < size; ++i){
        ref_vec_b[i] = b_vec1[i];
        ref_vec_b[i + size ] =  b_vec2[i];
    }

    ref_vec_out.reinit(2*size);
    laplace_matrix.reinit(sparsity_pattern);
    homogeneous_matrix.reinit(sparsity_pattern);
    damping_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);
    //
    MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(fe.degree + 1), mass_matrix);

    MatrixCreator::create_laplace_matrix(dof_handler, QGauss<dim>(fe.degree + 1), laplace_matrix);
    homogeneous_matrix.copy_from(laplace_matrix);
    homogeneous_matrix *= coef_a;
    damping_matrix.copy_from(laplace_matrix);
    damping_matrix *= coef_damping;

    const IdentityMatrix id_matrix(size);
    identity_m = id_matrix; 
    ref_matrix_func.reinit(2*size,2*size);
    first_index_set = linspace(0,size,1);
    second_index_set = linspace(size,2*size,1);

    identity_m.scatter_matrix_to(first_index_set,second_index_set,ref_matrix_func);

    FullMatrix<double> mass_matrix_full;
    mass_matrix_full.reinit(size,size);
    mass_matrix_full.copy_from(mass_matrix);
    mass_matrix_full.gauss_jordan();

    FullMatrix<double> homogeneous_matrix_full(size,size);
    FullMatrix<double> M_inv_homogeneous_matrix_full(size,size);
    homogeneous_matrix_full.copy_from(homogeneous_matrix);
    homogeneous_matrix_full *= -1.;
    //M_inv_homogeneous_matrix_full = homogeneous_matrix_full;
    mass_matrix_full.mmult(M_inv_homogeneous_matrix_full,homogeneous_matrix_full);
    M_inv_homogeneous_matrix_full.scatter_matrix_to(second_index_set,first_index_set,ref_matrix_func);
    
    FullMatrix<double> M_inv_damping_matrix_full(size,size);
    M_inv_damping_matrix_full = M_inv_homogeneous_matrix_full;
    M_inv_damping_matrix_full *= coef_damping/coef_a;
    M_inv_damping_matrix_full.scatter_matrix_to(second_index_set,second_index_set,ref_matrix_func);
}

template<int dim>   
void testChebprotype<dim>::create_ref_matrix_func(FullMatrix<double>& A, Vector<double>& b, Vector<double>& ref_vec_out)
{
        // temp variable 
        int two_size = 2*size; //2*size;
        const int lda = two_size;
        const int ldvl = two_size;
        const int ldvr = two_size;
        int info(0);
        std::vector<double> A_temp(two_size * two_size);        
        std::vector<double> wr(two_size), wi(two_size), vl(two_size*two_size), vr(two_size*two_size);        
        //
        eigenvectors_ref.reinit(two_size,two_size);
        eigenvectors_inverse_ref.reinit(two_size,two_size);
        eigenvalues.resize(two_size);
        for ( int i = 0; i < two_size; ++i) {
            for ( int j = 0; j < two_size; ++j) {
                A_temp[i * two_size + j] = A(i, j); 
            }
        }

        info = LAPACKE_dgeev(LAPACK_ROW_MAJOR,'N', 'V',two_size, A_temp.data() ,lda,  wr.data(), wi.data(), vl.data(), ldvl ,vr.data() ,ldvr);
        Assert (info >=0, ExcInternalError());
        if (info != 0)
            std::cerr << "LAPACK error in dgeev" << std::endl;

        // std::cout << "2 " ;
        for ( int i = 0; i < two_size; ++i){
            eigenvalues[i] = std::complex<double>( wr[i], wi[i] );
        }

        int col = 0;
        while (col < two_size) {
            if (wi[col] == 0.0) {  // Real eigenvalue
                for (int j = 0; j < two_size; ++j) {
                    eigenvectors_ref(j, col) = {vr[j * two_size + col], 0.0}; // dgeev saves eigenvectors row-wise
                }
                col++;
            } else {  // Complex pair (col and col+1)
                for (int j = 0; j < two_size; ++j) {
                    // Eigenvector for wr[col] + i·wi[col]
                    eigenvectors_ref(j, col) = {
                        vr[j * two_size + col],
                        vr[j * two_size + (col + 1)]  // dgeev saves eigenvectors row-wise
                    };
                    // Eigenvector for wr[col] - i·wi[col] (conjugate)
                    eigenvectors_ref(j, col + 1) = {
                        vr[j * two_size + col ],
                        -vr[j * two_size + (col + 1)] // dgeev saves eigenvectors row-wise
                    };
                }
                col += 2;  // Skip the next column (already handled)
            }
        }

        eigenvectors_inverse_ref.copy_from(eigenvectors_ref);
        eigenvectors_inverse_ref.gauss_jordan();

        std::complex<double> sum;
        for (int i=0; i<two_size; ++i)   {
            for (int j=0; j<two_size; ++j){
                sum = 0.;
                for (int l=0; l<two_size; ++l){
                    sum +=  eigenvectors_ref(i,l)*phi1( time_step_size*eigenvalues[l] )* eigenvectors_inverse_ref(l,j); 
                }
                A(i,j) = sum.real();
            }
        }   

        for (int i=0; i<two_size; ++i)   {
            for (int j=0; j<two_size; ++j){
                ref_vec_out[i] += A(i,j)*b[j];
            }
        } 
}

template<int dim>   
double testChebprotype<dim>::error_calc(Vector<double>& ref_vec_long, Vector<double>& calc_vec1, Vector<double>& calc_vec2)
{
    Vector<double> ref_vec1;
    Vector<double> ref_vec2;

    ref_vec1.reinit(size);
    ref_vec2.reinit(size);

    for (int i = 0; i < size; ++i){
        ref_vec1[i] = ref_vec_long[i];
        ref_vec2[i] = ref_vec_long[i+ size];
    }

    Vector<double> err_vec1 =  ref_vec1;
    err_vec1 -= calc_vec1;
    Vector<double> err_vec2 =  ref_vec2;
    err_vec2 -= calc_vec2;

    double err_total = std::sqrt( err_vec1.l2_norm()*err_vec1.l2_norm() + err_vec2.l2_norm()* err_vec2.l2_norm() );
    double scale =  std::sqrt(std::pow(ref_vec1.l2_norm(),2) + std::pow(ref_vec2.l2_norm(),2));
    return err_total/scale;
        
}

template<int dim>
double testChebprotype<dim>::run()
{
    const unsigned int initial_refinement    = 3;
    time_step_size                           = 0.5;
    // degree                                   = 100;
    tol                                      = 1e-4;


    GridGenerator::hyper_cube(triangulation,0,1);
    triangulation.refine_global(initial_refinement);
    setup_system();
    std::cout.setstate(std::ios_base::failbit);
    expoint.setup_system_macro_space();
    std::vector<int> config = expoint.find_opt_ellipse_prototype(ref_matrix_func);
    create_ref_matrix_func(ref_matrix_func,ref_vec_b,ref_vec_out);

    std::function<std::complex<double>(std::complex<double>)> func =  [](std::complex<double> z) {return phi1(z);};
    degree = expoint.estimate_degree_phi_k(1,tol,degree);
    // expoint.degree = degree;
    auto vmult = [this](FullMatrix<std::complex<double>> & Ma) { return this->matrix_matrix_product(Ma); };
    expoint.matrix_func_cheb(vmult,func,b_vec1,b_vec2,tol);
    std::cout.clear();
    return this->error_calc(ref_vec_out,b_vec1,b_vec2);
}

int main()
{
    const unsigned int dim = 1;
    using namespace dealii;
    using namespace MavesS;
    using namespace Data;
    using namespace Solving;

    double err; 

    testChebprotype<dim> test;
    err = test.run();
    if(err <  test.tol )
        std::cout<< 1 ;
    else
        std::cout<< 0 << "  error = "<< err ;
    // std::cout << std::endl;
    // std::cout << std::endl;
    // throw std::runtime_error("Something went wrong"); debugging
}

