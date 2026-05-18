#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <deal.II/base/utilities.h>

#include "../../algorithm/matrix_function.h"
#include "../../algorithm/global.h"

#include <iostream>
#include <cmath>
#include <string>
#include <chrono>
#include <vector>
#include <list>
#include <sstream>

namespace pt = boost::property_tree;

using namespace dealii;
using namespace MavesS;
using namespace TimeIntegration;

// ------------------------
// Helper: split string
// ------------------------
std::list<std::string> split(const std::string &input, char delimiter)
{
    std::list<std::string> result;
    std::stringstream ss(input);
    std::string item;

    while (std::getline(ss, item, delimiter))
        result.push_back(item);

    return result;
}

// ------------------------
// Templated runner
// ------------------------
template <int dim>
void run_example(int example)
{
    std::cout << "------Start simulation (dim = " << dim << ")------\n";

    auto start_time = std::chrono::steady_clock::now();

    pt::ini_parser::read_ini(
        "../../../config/example" + std::to_string(example) + ".ini",
        global::config);

    std::string mode_string =
        global::config.get<std::string>("Calculation_Mode.method_matrix_function");
    std::string beta_string =
        global::config.get<std::string>("Data.beta");

    auto mode_list = split(mode_string, ',');
    auto beta_list = split(beta_string, ',');

    std::cout << "methods = { ";
    for (const auto &m : mode_list)
        std::cout << m << " ";
    std::cout << "}\n";

    std::cout << "betas = { ";
    for (const auto &b : beta_list)
        std::cout << b << " ";
    std::cout << "}\n";

    MatrixFunction<dim> matrix_function_test;
    matrix_function_test.setup_system();

    std::vector<double> err_vector;
    err_vector.reserve(beta_list.size() * mode_list.size());

    for (const std::string &b : beta_list)
    {
        std::cout << "\n=============================================\n";
        std::cout << "Compute reference quantities for beta = " << b << "\n";
        std::cout << "=============================================\n";

        double beta_value = std::stod(b);

        matrix_function_test.setup_matrices(beta_value);
        matrix_function_test.expoint.setup_system_macro_space();

        if (global::config.get<bool>("Calculation_Mode.clc_ref_ellipse"))
            matrix_function_test.create_ref_ellipse();

        matrix_function_test.create_ref_matrix_func_poly_krylov();

        for (const std::string &mode_name : mode_list)
        {
            typename MatrixFunction<dim>::MethodMatrixFunction mode;

            if (mode_name == "poly_krylov")
                mode = MatrixFunction<dim>::poly_krylov;
            else if (mode_name == "rat_krylov")
                mode = MatrixFunction<dim>::rat_krylov;
            else if (mode_name == "cheb")
                mode = MatrixFunction<dim>::cheb;
            else
                throw std::runtime_error("Unknown method: " + mode_name);

            matrix_function_test.set_method(mode);
            matrix_function_test.run(b);

            double err, time;
            int mv;

            matrix_function_test.get_results(err, time, mv);
            err_vector.push_back(err);
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    int size;
    matrix_function_test.get_size(size);

    std::cout << "\n==============================================\n";
    std::cout << "Example:        "
              << global::config.get<std::string>("Calculation_Mode.example") << "\n";
    std::cout << "Method:         "
              << global::config.get<std::string>("Calculation_Mode.method_matrix_function") << "\n";
    std::cout << "Grid width:     2^-"
              << global::config.get<std::string>("Space_Discretization.initial_refinement") << "\n";
    std::cout << "FE degree:      "
              << global::config.get<std::string>("Space_Discretization.fe_degree") << "\n";
    std::cout << "Matrix size:    " << size << "\n";
    std::cout << "Epsilon:        2^"
              << global::config.get<std::string>("Data.epsilon") << "\n";
    std::cout << "Beta:           "
              << global::config.get<std::string>("Data.beta") << "\n";
    std::cout << "Total Time:     " << elapsed_seconds.count() << " s\n";
    std::cout << "==============================================\n";

    std::cout << "-------End simulation------\n";
}

// ------------------------
// main
// ------------------------
int main(int argc, char **argv)
{
    try
    {
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        std::vector<int> examples = {101, 201, 301};

        for (int example : examples)
        {
            if (example < 200)
                run_example<1>(example);
            else
                run_example<2>(example);
        }
    }
    catch (std::exception &exc)
    {
        std::cerr << "\nException:\n" << exc.what() << "\n";
        return 1;
    }
    catch (...)
    {
        std::cerr << "\nUnknown exception!\n";
        return 1;
    }

    return 0;
}