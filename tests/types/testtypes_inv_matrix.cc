#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/dofs/dof_tools.h>


#include "../../data/ExactSolution.h"
#include "../../data/InitialConditions.h"

#include "../../types/InverseMatrix.h"
#include "../../tools/Output.h"

#include <iostream>
#include <random>
#include <lapacke.h>
#include <complex>
#include <functional>
#include <chrono>


using namespace dealii;
using namespace MavesS;
using namespace Data;

template< typename VectorType>
void generateRandomVector(VectorType& vec, double min_val, double max_val) {
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<> dis(min_val, max_val);

    for (unsigned int i = 0; i < vec.size(); ++i) {
        vec[i] = dis(gen);  // Assign random value
    }
}

// template <typename T>
// std::vector<T> linspace_steps(T start, T end, int steps) {
//     std::vector<T> result;

//     for (int i = 0; i  < steps; ++i) {
//         result.push_back(start + i * (end - start)/steps);
//     }
//     return result;
// }

// template< typename VectorType>
// void generate_sinus(VectorType& vec, double min_val, double max_val) {
//     std::vector<double> x_values = linspace_steps(min_val, max_val, vec.size());
//     for (unsigned int i = 0; i < vec.size(); ++i) {
//         vec[i] = std::sin(x_values[i]);  // Assign sinusoidal value
//     }
// }

template <int dim, typename number = double>
    class generateVectorfromFunction : public Function<dim,number>
    {
    public:

        generateVectorfromFunction( const int example)
            : Function<dim,number>()
            ,example(example)
        {}

        number value(const Point<dim> & x,
                                const unsigned int component = 0) const override {

            (void)component;
            AssertIndexRange(component, 1);
            Assert(dim < 3, ExcNotImplemented());
            ///
            if(example == -1) {

                return sin(3.1415926535897931*pow(x[0], 2));

            }
            ///
            else if(example == -2){ 
                return  1.5707963267948966*x[0]*x[1]*(1 - x[0])*(1 - x[1]);
            }
                                        
        };
    private:
        const int example;

};


template <int dim, typename number = double>
class testInverseMatrix
{
    public:
        testInverseMatrix();
        double run(bool mass_lumping, bool with_preconditioner = false);
    private:
        void setup_system();
        void create_ref_vec( Vector<number> b, Vector<number>& ref_vec_out);
        double M_inv_vmult( Vector<number>& b, bool mass_lumping = false, bool with_preconditioner = false);
        double error_calc(Vector<number>& ref_vec, Vector<number>& calc_vec);

        Triangulation<dim> triangulation;

        DoFHandler<dim>    dof_handler;
        FE_Q<dim>          fe; 
        SparsityPattern    sparsity_pattern;

        SparseMatrix<double>    mass_matrix_r;
        SparseMatrix<number>    mass_matrix;

        AffineConstraints<number> constraints;
        // random vectors
        Vector<number> b_vec; 
        Vector<number> Minv_b_ref; 
        //
        int    size;
        std::map<types::global_dof_index, number> boundary_values0;
        std::map<types::global_dof_index, number> boundary_values1;


};

template<int dim , typename number>
testInverseMatrix<dim,number>::testInverseMatrix()
:   
    dof_handler(triangulation)
   ,fe(1)

{}

template<int dim , typename number>
void testInverseMatrix<dim,number>::setup_system()
{

    dof_handler.distribute_dofs(fe);
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();
    size = dof_handler.n_dofs();
    //
    b_vec.reinit(size);
    Vector<double> b_vec_real(size);
    if(dim == 2){    
        VectorTools::interpolate(dof_handler,
                                generateVectorfromFunction<dim>(-2),
                                b_vec_real);}
    else if(dim == 1){
        VectorTools::interpolate(dof_handler,
                                generateVectorfromFunction<dim>(-1),
                                b_vec_real);}
    else
    {
        std::cout << "Error: Dimension not supported!" << std::endl;
        exit(1);
    }
    for (unsigned int i = 0; i < size; ++i) {
        b_vec[i] = b_vec_real[i];
    }
    Minv_b_ref.reinit(size);
    //
    DynamicSparsityPattern dsp(size);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    mass_matrix_r.reinit(sparsity_pattern);
    MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(fe.degree +1), mass_matrix_r);
    for (unsigned int i=0; i<mass_matrix_r.m(); ++i)
        for (SparseMatrix<double>::const_iterator it = mass_matrix_r.begin(i); it != mass_matrix_r.end(i); ++it)
        {
            mass_matrix.set(it->row(), it->column(), number(it->value()));
        }
    mass_matrix.compress(VectorOperation::insert);
    
}

template<int dim , typename number>   
void testInverseMatrix<dim,number>::create_ref_vec( Vector<number> b, Vector<number>& ref_vec_out)
{
    SparseMatrix<double> mass_matrix_ref_double;
    mass_matrix_ref_double.reinit(sparsity_pattern);
    MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(fe.degree +1), mass_matrix_ref_double);
    SparseMatrix<number> mass_matrix_ref;
    mass_matrix_ref.reinit(sparsity_pattern);
    for (unsigned int i=0; i<mass_matrix_ref_double.m(); ++i)
        for (SparseMatrix<double>::const_iterator it = mass_matrix_ref_double.begin(i); it != mass_matrix_ref_double.end(i); ++it)
        {
            mass_matrix_ref.set(it->row(), it->column(), number(it->value()));
        }
    mass_matrix_ref.compress(VectorOperation::insert); 
    //
    VectorTools::interpolate_boundary_values(dof_handler, 0,
                                                Functions::ConstantFunction<dim,number>(0.0), boundary_values0);
    // for (unsigned int i=0; i<mass_matrix_ref.m(); ++i)
    //     for (SparseMatrix<double>::const_iterator it = mass_matrix_ref.begin(i); it != mass_matrix_ref.end(i); ++it)
    //     {
    //     std::cout << " i " << it->row() << " j " << it->column()  << " x " << number(it->value()) ;
    // }
    MatrixTools::apply_boundary_values(boundary_values0, mass_matrix_ref, ref_vec_out, b);
    
    if(dim == 1){
        VectorTools::interpolate_boundary_values(dof_handler, 1,
            Functions::ConstantFunction<dim,number>(0.0), boundary_values1);
            
            MatrixTools::apply_boundary_values(boundary_values1, mass_matrix_ref, ref_vec_out, b);
    }
    // for (unsigned int i=0; i<mass_matrix_ref.m(); ++i)
    //     for (SparseMatrix<double>::const_iterator it = mass_matrix_ref.begin(i); it != mass_matrix_ref.end(i); ++it)
    //     {
    //     std::cout << " i " << it->row() << " j " << it->column()  << " y " << number(it->value()) ;
    // }

    //
    FullMatrix<number> mass_matrix_full;
    mass_matrix_full.reinit(size,size);
    mass_matrix_full.copy_from(mass_matrix_ref);
    mass_matrix_full.gauss_jordan();

        for (int i=0; i<size; ++i)   {
            for (int j=0; j<size; ++j){
                ref_vec_out[i] += mass_matrix_full(i,j)*b[j];
            }
        }   

}

template<int dim , typename number>   
double testInverseMatrix<dim,number>::error_calc(Vector<number>& ref_vec, Vector<number>& calc_vec)
{
    Vector<number> err_vec =  ref_vec;
    err_vec -= calc_vec;
    
    double err_total = err_vec.l2_norm();
    return err_total / ref_vec.l2_norm();    
    
}

template<int dim , typename number>
double testInverseMatrix<dim,number>::M_inv_vmult( Vector<number>& b, bool mass_lumping, bool with_preconditioner)
{
    Vector<number> dst;
    dst.reinit(size);
    int last_Step = 0;
    int numb_iter = 100; // Number of iterations for timing
    std::vector<double> durations; // typo fixed: "dourations" -> "durations"
    std::chrono::microseconds total_duration(0);
    std::chrono::high_resolution_clock::time_point start, end;

    if(mass_lumping){
        Types::InverseMatrix<dim,number> M_inv(mass_matrix,  dof_handler);
        for (unsigned int i = 0; i < numb_iter; ++i) {
            dst = 0.; // Reset dst vector for each iteration
            if (with_preconditioner){
                start = std::chrono::high_resolution_clock::now();
                last_Step = M_inv.vmult_mass_lumping_preconditioner(dst, b);
                end = std::chrono::high_resolution_clock::now();
            } else {
                start = std::chrono::high_resolution_clock::now();
                last_Step = M_inv.vmult_mass_lumping(dst, b);
                end = std::chrono::high_resolution_clock::now();
            }

            // Append duration in microseconds
            durations.push_back(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            );
        }

        // Compute mean duration
        for (const auto& d : durations) {
            total_duration += std::chrono::microseconds(static_cast<long long>(d));
        }
        auto mean_duration = total_duration / durations.size();
        if(with_preconditioner){
            std::cout << "      Average time for M_inv.vmult_mass_lumping_preconditioner: "
                << mean_duration.count() << " microseconds" ;
            std::cout << "Last step in M_inv.vmult: " << last_Step<< " ";
        }
        else if (!with_preconditioner){
            std::cout << "      Average time for M_inv.vmult_mass_lumping: "
                << mean_duration.count() << " microseconds" ;}

    } else if(with_preconditioner) {
        Types::InverseMatrix<dim,number> M_inv(mass_matrix,  dof_handler);
        for (unsigned int i = 0; i < numb_iter; ++i) {
            dst = 0.; // Reset dst vector for each iteration
            start = std::chrono::high_resolution_clock::now();
            last_Step = M_inv.vmult_with_preconditioner(dst, b);
            end = std::chrono::high_resolution_clock::now();

            // Append duration in microseconds
            durations.push_back(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            );
        }

        // Compute mean duration
        for (const auto& d : durations) {
            total_duration += std::chrono::microseconds(static_cast<long long>(d));
        }
        auto mean_duration = total_duration / durations.size();

        std::cout <<  "      Average time for M_inv.vmult_with_preconditioner: "
                << mean_duration.count() << " microseconds" ;
        std::cout << "Last step in M_inv.vmult: " << last_Step<< " ";

    } else {
        Types::InverseMatrix<dim,number> M_inv(mass_matrix, dof_handler);
        for (unsigned int i = 0; i < numb_iter; ++i) {
            dst = 0.; // Reset dst vector for each iteration
            start = std::chrono::high_resolution_clock::now();
            last_Step = M_inv.vmult_without_preconditioner(dst, b);
            end = std::chrono::high_resolution_clock::now();

            // Append duration in microseconds
            durations.push_back(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            );
        }

        // Compute mean duration
        for (const auto& d : durations) {
            total_duration += std::chrono::microseconds(static_cast<long long>(d));
        }
        auto mean_duration = total_duration / durations.size();

        std::cout << "      Average time for M_inv.vmult_without_precondition: "
                << mean_duration.count() << " microseconds" ;
        std::cout << "Last step in M_inv.vmult_without_precondition: " << last_Step<< " ";
    }
    return this->error_calc(dst, Minv_b_ref);
}

template<int dim , typename number>
double testInverseMatrix<dim,number>::run( bool mass_lumping, bool with_preconditioner)
{
    triangulation.clear(); // Fix: ensure empty before re-creating
    const unsigned int initial_refinement    = 5;

    GridGenerator::hyper_cube(triangulation,0,1);
    triangulation.refine_global(initial_refinement);
    setup_system();
    create_ref_vec(b_vec,Minv_b_ref);
    return M_inv_vmult(b_vec, mass_lumping, with_preconditioner);
}

int main()
{
    const unsigned int dim = 1;
    const double ERROR_THRESHOLD = 0.1;
    using namespace dealii;
    using namespace MavesS;

    testInverseMatrix<dim,double> test;

    std::cout.setstate(std::ios_base::failbit);
    double err1 = test.run(true,false); // with mass lumping
    double err2 = test.run(true,true); // with mass lumping preconditioner
    double err3 = test.run(false,false); // without preconditioner
    double err4 = test.run(false,true); // with preconditioner
    std::cout.clear();

    if(err1 <  ERROR_THRESHOLD && err2 <  ERROR_THRESHOLD && err3 <  ERROR_THRESHOLD)
        std::cout<< 1 ;
    else
        std::cout<< 0 << "  error " << err1 << " " << err2 << " " << err3 << " " << err4 << std::endl;
}

