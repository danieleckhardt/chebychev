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
#include "../../types/BlockMatrix.h"

#include "../../time_integration/expoInt_tools/expoInt.h"
#include "../../time_integration/expoInt_tools/chebyshev_approx.h"
#include "../../time_integration/expoInt_tools/calc_eigen.h"
#include "../../tools/Output.h"

#include <iostream>
#include <random>
#include <lapacke.h>
#include <complex>
#include <functional>
#include <slepceps.h> 


using namespace dealii;
using namespace MavesS;
using namespace Data;
using namespace TimeIntegration;

// template< typename VectorType>
// void generateRandomVector(VectorType& vec, double min_val, double max_val) {
//     std::random_device rd;  // Seed
//     std::mt19937 gen(rd()); // Mersenne Twister engine
//     std::uniform_real_distribution<> dis(min_val, max_val);

//     for (unsigned int i = 0; i < vec.size(); ++i) {
//         vec[i] = dis(gen);  // Assign random value
//     }
// }


// template< typename VectorType>
// void generate_sinus(VectorType& vec, double min_val, double max_val) {
//     std::vector<double> x_values = linspace_steps(min_val, max_val, vec.size());
//     for (unsigned int i = 0; i < vec.size(); ++i) {
//         vec[i] = std::sin(x_values[i]);  // Assign sinusoidal value
//     }
// }
// template< typename VectorType>
// void generate_cosinus(VectorType& vec, double min_val, double max_val) {
//     std::vector<double> x_values = linspace_steps(min_val, max_val, vec.size());
//     for (unsigned int i = 0; i < vec.size(); ++i) {
//         vec[i] = std::cos(x_values[i]);  // Assign sinusoidal value
//     }
// }

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
class testfindoptellipse
{
    public:
        testfindoptellipse();
        double run();
        void set_coefficients(double a, double d) {  coef_a = a;   coef_damping = d;};
        void set_triangulation();
        double tol;
    private:
        void setup_system();
        double error_calc(double& ref_radiusx, double& ref_radiusy, double& ref_center, std::complex<double>& ref_eccentricity);


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

        // setup blockmatrix
        std::vector<int> first_index_set;
        std::vector<int> second_index_set;
        //
        std::vector<std::complex<double>> eigenvalues;
        FullMatrix<std::complex<double>> eigenvectors;

        double time_step_size;
        double coef_a  ;
        double coef_damping ;
        int    size;
        int    degree;

        ExpoInt<dim> expoint;
        Output<dim,FullMatrix<double>> output_class;
        Output<dim,FullMatrix<std::complex<double>>> output_class_complex;
        Calc_eigen<dim> calc_eigen;


};

template<int dim>
testfindoptellipse<dim>::testfindoptellipse()
:   
    dof_handler(triangulation)
   ,fe(1)
   ,expoint(fe, dof_handler,constraints,homogeneous_matrix,damping_matrix,mass_matrix,time_step_size,0)
   ,calc_eigen(eigenvalues,eigenvectors)

{}

template<int dim>
void testfindoptellipse<dim>::setup_system() {
    
    //
    dof_handler.distribute_dofs(fe);
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();
    size = dof_handler.n_dofs();
    DynamicSparsityPattern dsp(size);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
    sparsity_pattern.copy_from(dsp);
    
    laplace_matrix.reinit(sparsity_pattern);
    homogeneous_matrix.reinit(sparsity_pattern);
    damping_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);
    //
    MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(fe.degree + 1), mass_matrix);
    // Apply b.c. on mass_matrix
    Vector<double> dst{};
    Vector<double> src{};
    dst.reinit(dof_handler.n_dofs());
    src.reinit(dof_handler.n_dofs());
    dst = 0.; src = 0.;
    std::map<types::global_dof_index,double> boundary_values0;
    std::map<types::global_dof_index,double> boundary_values1;
    // boundary conditions for mass matrix
    VectorTools::interpolate_boundary_values(dof_handler, 0,
                                                Functions::ConstantFunction<dim>(0.0), boundary_values0);
    MatrixTools::apply_boundary_values(boundary_values0, mass_matrix, dst, src);

    if (dim == 1) {
        VectorTools::interpolate_boundary_values(dof_handler, 1,
                                                    Functions::ConstantFunction<dim>(0.0), boundary_values1);
        MatrixTools::apply_boundary_values(boundary_values1, mass_matrix, dst, src);
    }

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
double testfindoptellipse<dim>::error_calc(double& ref_radiusx, double& ref_radiusy, double& ref_center, std::complex<double>& ref_eccentricity)
{

    double err_radiusx,err_radiusy, err_center;
    std::complex<double> err_eccentricity;
    calc_eigen.calc_all_eig(ref_matrix_func,2*size);
    double max_rad = 0.;
    double ref_radius_ellipse;
    expoint.ellipse_opt.get_radius_ellipse(ref_radius_ellipse);
    expoint.ellipse_opt.get_parameters(err_radiusx,err_radiusy,err_eccentricity,err_center);
    std::complex<double> bd_point;
    if(err_eccentricity.imag() == 0){
        bd_point = std::complex<double>(err_radiusx + err_center);
    }
    else{
        bd_point = std::complex<double>(err_radiusy*1i + err_center);
    }
    for (int i = 0; i < eigenvalues.size(); ++i) {
        eigenvalues[i] *= time_step_size;
        // std::cout << "Eigenvalue " << i << ": " << eigenvalues[i] << " Radius: " << expoint.ellipse_opt.radius_p(eigenvalues[i]) << "; " ;
        if (max_rad < expoint.ellipse_opt.radius_p(eigenvalues[i])* 1./expoint.ellipse_opt.radius_p(bd_point ) ){
            max_rad = expoint.ellipse_opt.radius_p(eigenvalues[i])* 1./expoint.ellipse_opt.radius_p(bd_point ) ;
        }
    }
    // std::cout << "max_rad:" << max_rad << ", ell_rad" << ref_radius_ellipse << std::endl;
    // std::cout<< " xr " <<ref_radiusx << " " <<  err_radiusx<< std::endl;
    // std::cout<< " yr " <<ref_radiusy << " " <<  err_radiusy<< std::endl;
    // std::cout<< " cr " <<ref_center << " " <<  err_center<< std::endl;
    // std::cout<< " er " <<ref_eccentricity << " " <<  err_eccentricity<< std::endl;
    err_radiusx     -= ref_radiusx;
    err_radiusy     -= ref_radiusy;
    err_center      -= ref_center;
    err_eccentricity -= ref_eccentricity;


    double err_total = std::pow(std::abs(err_radiusx),2) + std::pow(std::abs(err_radiusy),2) + std::pow(std::abs(err_center),2) + std::pow(std::abs(err_eccentricity),2);
    double scale     = std::pow(std::abs(ref_radiusx),2) + std::pow(std::abs(ref_radiusy),2) + std::pow(std::abs(ref_center),2) + std::pow(std::abs(ref_eccentricity),2);
    return std::sqrt(err_total/scale);
        
}

template<int dim>
void testfindoptellipse<dim>::set_triangulation(){
    const unsigned int initial_refinement    = 5;
    GridGenerator::hyper_cube(triangulation,0,1);
    triangulation.refine_global(initial_refinement);
}

template<int dim>
double testfindoptellipse<dim>::run()
{
    time_step_size                           = 1;
    degree                                   = 1;
    tol                                      = 1e-4;

    setup_system();
    std::cout.setstate(std::ios_base::failbit);
    expoint.setup_system_macro_space();
    std::vector<int> config = expoint.find_opt_ellipse_prototype(ref_matrix_func);
    //std::cout << "Config" << config[0] << " " << config[1] << " " << config[2] << " ";
    double ref_radiusx, ref_radiusy, ref_center;
    std::complex<double> ref_eccentricity;
    expoint.ellipse_opt.get_parameters(ref_radiusx,ref_radiusy,ref_eccentricity,ref_center);
    // std::cout << "Reference Ellipse: radiusx = " << ref_radiusx 
    //           << ", radiusy = " << ref_radiusy 
    //           << ", center = " << ref_center 
    //           << ", eccentricity = " << ref_eccentricity 
    //           << " ";     
    Types::BlockMatrix<dim> blockmatrixtest(homogeneous_matrix,damping_matrix,mass_matrix,dof_handler, time_step_size, false);
    expoint.find_opt_ellipse(tol, &blockmatrixtest);
    std::cout.clear();
    return this->error_calc(ref_radiusx,ref_radiusy,ref_center,ref_eccentricity);
}

int main(int argc, char **argv)
{
    SlepcInitialize(&argc, &argv, NULL, NULL);
    const unsigned int dim = 1;
    using namespace dealii;
    using namespace MavesS;
    using namespace Data;
    using namespace Solving;

    double err; 
    bool TEST_STILL_CORRECT = true; 


    testfindoptellipse<dim> test;
    test.set_triangulation();
    
    test.set_coefficients(0.5, 0.01); // Set coefficients for the test
    err = test.run();
    if(err >  test.tol ){
        std::cerr << "Test 1 failed with error: " << err ;
        TEST_STILL_CORRECT = false;
    }
    //
    test.set_coefficients(0.5, 0.4); // Set coefficients for the test
    err = test.run();
    if(err >  test.tol ){
        std::cerr << "Test 2 failed with error: " << err ;
        TEST_STILL_CORRECT = false;
    }
    //
    test.set_coefficients(0.5, 0.001); // Set coefficients for the test
    err = test.run();
    if(err >  test.tol ){
        std::cerr << "Test 3 failed with error: " << err ;
        TEST_STILL_CORRECT = false;
    }
    //
    // Output the result
    //
    if(TEST_STILL_CORRECT)
        std::cout<< 1;
    else
        std::cout<< 0 ;

     SlepcFinalize();
    //  std::cout << " " << std::endl;
    //  throw 42; for debugging
     return 0;

}

