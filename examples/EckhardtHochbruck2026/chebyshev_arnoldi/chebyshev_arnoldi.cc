#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/foreach.hpp>
#include <deal.II/base/utilities.h>


#include "../../../algorithm/matrix_function.h"
#include "../../../algorithm/global.h"

#include <iostream>
#include <cmath>
#include <string>
#include <chrono> // Walltime

namespace pt = boost::property_tree;
int main(int argc, char **argv)
{
// SlepcInitialize(&argc, &argv, NULL, NULL); 
try
    {
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        auto start_time = std::chrono::steady_clock::now();
        using namespace dealii;
        using namespace MavesS;
        using namespace TimeIntegration;


        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///////

        const int example = 101; // choose the Example
        const unsigned int dim = 1; // if the example number = 1x, than set dim = 1, if example number = 2x, than  dim = 2

        ///////// Usually you do not have to change anything above the line
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        std::cout<< "------Start simulation------" << std::endl;
        pt::ini_parser::read_ini("../../../../config/example"+ std::to_string(example) + ".ini", MavesS::global::config);
        std::vector<double> err_vector;

        MatrixFunction<dim>::MethodMatrixFunction mode; 
        std::string mode_string = global::config.get<std::string>("Calculation_Mode.method_matrix_function");
        std::string beta_string = global::config.get<std::string>("Data.beta");

        std::string delimiter = ",";

        size_t pos = 0;
        std::list<std::string> mode_list;
        std::list<std::string> beta_list;
        while ((pos = mode_string.find(delimiter)) != std::string::npos) {
            mode_list.push_back(mode_string.substr(0, pos));
            mode_string.erase(0, pos + delimiter.length());
        }
        pos = 0;
        while ((pos = beta_string.find(delimiter)) != std::string::npos) {
            beta_list.push_back(beta_string.substr(0, pos));
            beta_string.erase(0, pos + delimiter.length());
        }
        mode_list.push_back(mode_string.substr(0, pos));
        beta_list.push_back(beta_string.substr(0, pos));
            std::cout << "methods = { ";
        for (std::string n : mode_list)
            std::cout << n << ", ";
        std::cout << "};\n";
            std::cout << "betas = { ";
        for (std::string n : beta_list)
            std::cout << n << ", ";
        std::cout << "};\n";
        MatrixFunction<dim> matrix_function_test;
        err_vector.resize(beta_list.size()*mode_list.size());
        matrix_function_test.setup_system();

        for( std::string b : beta_list )
        {
          std::cout << std::endl;
          std::cout << "#############################################" << std::endl;
          std::cout << "#############################################" << std::endl;
          std::cout << "Compute reference quantities for beta = " << b<< std::endl;
          std::cout << "#############################################" << std::endl;
          std::cout << "#############################################" << std::endl;
          std::cout << std::endl;

          double beta_value = std::stod(b);
          matrix_function_test.setup_matrices(beta_value);
          matrix_function_test.expoint.setup_system_macro_space();
          if(global::config.get<bool>("Calculation_Mode.clc_ref_ellipse")){
            matrix_function_test.create_ref_ellipse();
          }
          matrix_function_test.create_ref_matrix_func();
          for(std::string mode_string : mode_list)
          {
                double err_temp; 
                double computation_time_temp;
                int mv_temp; 
                if(mode_string == "poly_krylov")
                  mode = MatrixFunction<dim>::poly_krylov;
                else if(mode_string == "rat_krylov") 
                  mode = MatrixFunction<dim>::rat_krylov;
                else if(mode_string == "cheb") 
                  mode = MatrixFunction<dim>::cheb;
                else
                {
                Assert(false, ExcNotImplemented());
                  mode = MatrixFunction<dim>::poly_krylov;
                }
                matrix_function_test.set_method(mode);
                matrix_function_test.run(b);
                matrix_function_test.get_results(err_temp, computation_time_temp, mv_temp);
                err_vector.push_back(err_temp);
            }
        }
    std::cout<< "-------End simulation------" << std::endl;

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    int size;
    matrix_function_test.get_size(size);
    std::cout 
      << "==============================================" 
      << std::endl
      << "   Example:                      " << global::config.get<std::string>("Calculation_Mode.example")
      << std::endl
      << "   Method:                       " << global::config.get<std::string>("Calculation_Mode.method_matrix_function")
      << std::endl
      << "   Grid width:                   2^-" << global::config.get<std::string>("Space_Discretization.initial_refinement")
      << std::endl
      << "   FE degree:                    " << global::config.get<std::string>("Space_Discretization.fe_degree")
      << std::endl
      << "   Matrix size:                  " << size
      << std::endl
      << "   Epsilon:                      2^" << global::config.get<std::string>("Data.epsilon")
      << std::endl
      << "   Beta:                         " << global::config.get<std::string>("Data.beta")
      << std::endl
      << "   Total Time:                   " << elapsed_seconds.count() << " s"
      << std::endl
      << "==============================================" 
      << std::endl;
    std::cout << "Path to Logfile:" << "../output/LastRun.txt" << std::endl;
  }
  catch (std::exception &exc)
    {
      std::cerr << std::endl 
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      // SlepcFinalize();
      return 1;   

    }

  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      // SlepcFinalize();
      return 1;
    }
  // SlepcFinalize();
  return 0;
}