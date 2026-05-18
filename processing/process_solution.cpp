#include "process_solution.h"

using namespace dealii;

namespace MavesS // Multi-Waves-Scale
{
using namespace Data;
namespace processing
{

template <int dim> 
Process_Solutions<dim>::Process_Solutions(DoFHandler<dim>&  dof_handler_macro, DoFHandler<dim>&  dof_handler_reference, Triangulation<dim> & triangulation_macro, Triangulation<dim> & triangulation_reference,  TimerOutput  &    computing_timer)
:
 triangulation_reference(triangulation_reference)
,triangulation_macro(triangulation_macro)
,dof_handler_macro(dof_handler_macro)
,dof_handler_reference(dof_handler_reference)
,computing_timer(computing_timer)
{}

template <int dim> 
void Process_Solutions<dim>::save_referencesolution(const Vector<double> &solution,const double &time, const bool &derivative_u, const bool &nonlinear, 
                                                                const bool &damped,  std::string example, std::string coefficient_type, unsigned int level_ref, std::string currentPath ) const{
    TimerOutput::Scope t(computing_timer, "save_referencesolution");
    std::string str_ref_path = currentPath + "/output/ref_solution/Example"+ example +"/";
    std::string str_tria_path = str_ref_path + "reference_tria";
    std::string str_config_path = str_ref_path + "config.ini";
    std::string str_ref;
    if(nonlinear)
    {
      str_ref = Utilities::int_to_string(dim) + "Dref_sol"; 
    }
    else
      str_ref = Utilities::int_to_string(dim) + "Dref_sollin";

    if(derivative_u)
      str_ref += "v";
    else
      str_ref += "u";

    if(damped)
    {
      str_ref += "damp"; 
    }

    std::string filename_ref_path;
    if(coefficient_type == "origin")
      filename_ref_path = str_ref_path + str_ref + "-"+std::to_string(level_ref) + "_origin"+ "_" + std::to_string(time);
    else if(coefficient_type == "hmm_hom")
      filename_ref_path = str_ref_path + str_ref + "-"+std::to_string(level_ref) + "_hmm_hom" + "_" + std::to_string(time);
    else if(coefficient_type == "exact_hom")
      filename_ref_path = str_ref_path + str_ref + "-"+std::to_string(level_ref) + "_exact_hom" "_" + std::to_string(time);

    Vector<double> solution_ref_out(solution);

    std::ofstream output_ref(filename_ref_path.c_str());

    solution_ref_out.block_write(output_ref);

}

template <int dim> 
void Process_Solutions<dim>::load_referencesolution(const double &time, const bool &derivative_u, const bool &nonlinear, const bool &damped, std::string example,  
                              std::string coef_type, unsigned int level_ref, Vector<double> & solution_ref, std::string currentPath ) {
    TimerOutput::Scope t(computing_timer, "load_referencesolution");
    std::string str_ref_path =  currentPath + "/output/ref_solution/Example" + example +"/";
    std::string str_ref;
    if(nonlinear)
      str_ref = Utilities::int_to_string(dim) +"Dref_sol";
    else
      str_ref = Utilities::int_to_string(dim) +"Dref_sollin";
    if(derivative_u)
      str_ref += "v";
    else
      str_ref += "u";

    if(damped)
    {
      str_ref += "damp"; 
    }
    std::string filename_ref_path = str_ref_path +  str_ref + "-" + std::to_string(level_ref) + "_" + coef_type+"_" +  std::to_string(time);
    std::cout << filename_ref_path << std::endl;
    std::ifstream input_ref(filename_ref_path.c_str());
    if (!input_ref)
      throw std::runtime_error("Reference solution file not found: " + filename_ref_path);
    solution_interpolat.reinit(dof_handler_reference.n_dofs());
    solution_ref.reinit(dof_handler_reference.n_dofs());
    solution_ref.block_read(input_ref);
}

template <int dim>
template < typename parametertype>
void Process_Solutions<dim>::process_exactsolution(Vector<double> &solution, Solution<dim> & exact_solution, ConvergenceTable &  convergence_table_L2,  
                                                    const unsigned int fe_degree, std::string & convergence_parameter,const parametertype convergence_resolution ){
    TimerOutput::Scope t(computing_timer, "process_exactsolution");

    Vector<float> difference_per_cell(triangulation_macro.n_active_cells());

    VectorTools::integrate_difference(dof_handler_macro,
                                  solution,
                                  exact_solution,
                                  difference_per_cell,
                                  QGauss<dim>(fe_degree + 1), 
                                  VectorTools::L2_norm);
    const double L2_error =
    VectorTools::compute_global_error(triangulation_macro,
                                    difference_per_cell,
                                    VectorTools::L2_norm);

    convergence_table_L2.add_value(convergence_parameter,convergence_resolution);
    convergence_table_L2.add_value("L2", L2_error);
}

template <int dim>
template < typename parametertype>
void Process_Solutions<dim>::process_exactsolution(Vector<double> &solution, Solution<dim> & exact_solution,  Vector<double> &solution_prime, Solution_prime<dim> & exact_solution_prime, 
                                                    ConvergenceTable &  convergence_table_L2, ConvergenceTable &  convergence_table_H1, ConvergenceTable &  convergence_table_total,  
                                                     const unsigned int fe_degree, std::string & convergence_parameter,const parametertype convergence_resolution ){
    TimerOutput::Scope t(computing_timer, "process_exactsolution");

    Vector<double> Zerovector{};

    Zerovector.reinit(dof_handler_macro.n_dofs());


    Vector<float> difference_per_cell(triangulation_macro.n_active_cells());
    VectorTools::integrate_difference(dof_handler_macro,
                                  solution_prime,
                                  exact_solution_prime,
                                  difference_per_cell,
                                  QGauss<dim>(fe_degree + 1), 
                                  VectorTools::L2_norm);
    double L2_error =
    VectorTools::compute_global_error(triangulation_macro,
                                    difference_per_cell,
                                    VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler_macro,
                                  solution,
                                  exact_solution,
                                  difference_per_cell,
                                  QGauss<dim>(fe_degree + 1), 
                                  VectorTools::H1_norm);
    double H1_error =
    VectorTools::compute_global_error(triangulation_macro,
                                    difference_per_cell,
                                    VectorTools::H1_norm);
    
    const double total_error = H1_error + L2_error;

    VectorTools::integrate_difference(dof_handler_macro,
                                  Zerovector,
                                  exact_solution_prime,
                                  difference_per_cell,
                                  QGauss<dim>(fe_degree + 1), 
                                  VectorTools::L2_norm);
    double L2_error_sol =
    VectorTools::compute_global_error(triangulation_macro,
                                    difference_per_cell,
                                    VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler_macro,
                                  Zerovector,
                                  exact_solution,
                                  difference_per_cell,
                                  QGauss<dim>(fe_degree + 1), 
                                  VectorTools::H1_norm);
    double H1_error_sol =
    VectorTools::compute_global_error(triangulation_macro,
                                    difference_per_cell,
                                    VectorTools::H1_norm);

    
    const double relativ_error = total_error/(H1_error_sol + L2_error_sol);

    convergence_table_L2.add_value(convergence_parameter,convergence_resolution);
    convergence_table_L2.add_value("L2", L2_error);
    convergence_table_H1.add_value(convergence_parameter,convergence_resolution);
    convergence_table_H1.add_value("H1", H1_error);
    convergence_table_total.add_value(convergence_parameter,convergence_resolution);
    convergence_table_total.add_value("H1+L2", relativ_error);
    
    convergence_table_H1.set_precision(convergence_parameter, 9);
    convergence_table_L2.set_precision(convergence_parameter, 9);
    convergence_table_total.set_precision(convergence_parameter, 9);
}

template <int dim>
template < typename parametertype>
void Process_Solutions<dim>::process_refsolution(Vector<double> &solution, Vector<double> &solution_ref, ConvergenceTable &  convergence_table_L2,  
                                                    const unsigned int fe_degree, std::string & convergence_parameter,const parametertype convergence_resolution ){
    TimerOutput::Scope t(computing_timer, "process_refsolution");

    Vector<double> difference_per_cell(triangulation_reference.n_active_cells());
    VectorTools::interpolate_to_different_mesh(dof_handler_macro,solution, dof_handler_reference,
                                                  solution_interpolat);
    Vector<double> diff_u = solution_ref;
    diff_u -= solution_interpolat;
    VectorTools::integrate_difference(dof_handler_reference,
                                      diff_u,
                                      Functions::ZeroFunction<dim>(),
                                      difference_per_cell,
                                      QGauss<dim>(fe_degree + 1), 
                                      VectorTools::L2_norm);
    const double L2_error =
    VectorTools::compute_global_error(triangulation_reference,
                                      difference_per_cell,
                                      VectorTools::L2_norm);
    convergence_table_L2.add_value(convergence_parameter,convergence_resolution);
    convergence_table_L2.add_value("L2", L2_error);
}

template <int dim>
template < typename parametertype>
void Process_Solutions<dim>::process_refsolution(Vector<double> &solution,Vector<double> &solution_ref,  Vector<double> &solution_prime, Vector<double> &solution_ref_prime, 
                                                    ConvergenceTable &  convergence_table_L2, ConvergenceTable &  convergence_table_H1, ConvergenceTable &  convergence_table_total,  
                                                     const unsigned int fe_degree, std::string & convergence_parameter,const parametertype convergence_resolution ){
    TimerOutput::Scope t(computing_timer, "process_refsolution");

    Vector<double> difference_per_cell(triangulation_reference.n_active_cells());
    VectorTools::interpolate_to_different_mesh(dof_handler_macro,solution_prime, dof_handler_reference,
                                                  solution_interpolat);
    Vector<double> diff_v = solution_ref_prime;
    diff_v -= solution_interpolat;
    VectorTools::integrate_difference(dof_handler_reference,
                                      diff_v,
                                      Functions::ZeroFunction<dim>(),
                                      difference_per_cell,
                                      QGauss<dim>(fe_degree + 1), 
                                      VectorTools::L2_norm);
    const double L2_error =
    VectorTools::compute_global_error(triangulation_reference,
                                      difference_per_cell,
                                      VectorTools::L2_norm);

    VectorTools::interpolate_to_different_mesh(dof_handler_macro,solution, dof_handler_reference,
                                                  solution_interpolat);
    Vector<double> diff_u = solution_ref;
    diff_u -= solution_interpolat;
    VectorTools::integrate_difference(dof_handler_reference,
                                      diff_u,
                                      Functions::ZeroFunction<dim>(),
                                      difference_per_cell,
                                      QGauss<dim>(fe_degree + 1), 
                                      VectorTools::H1_norm);
    const double H1_error =
    VectorTools::compute_global_error(triangulation_reference,
                                      difference_per_cell,
                                      VectorTools::H1_norm);

    const double total_error = H1_error + L2_error;

    convergence_table_L2.add_value(convergence_parameter,convergence_resolution);
    convergence_table_L2.add_value("L2", L2_error);
    convergence_table_H1.add_value(convergence_parameter,convergence_resolution);
    convergence_table_H1.add_value("H1", H1_error);
    convergence_table_total.add_value(convergence_parameter,convergence_resolution);
    convergence_table_total.add_value("H1+L2", total_error);

    convergence_table_H1.set_precision(convergence_parameter, 9);
    convergence_table_L2.set_precision(convergence_parameter, 9);
    convergence_table_total.set_precision(convergence_parameter, 9);
}
  
/// 
template class Process_Solutions<1>;
template class Process_Solutions<2>;
///
template void Process_Solutions<1>::process_refsolution<int>(Vector<double> &solution, Vector<double> &solution_ref, ConvergenceTable &  convergence_table_L2,  
                                                    const unsigned int fe_degree, std::string & convergence_parameter,const int convergence_resolution );
template void Process_Solutions<2>::process_refsolution<int>(Vector<double> &solution, Vector<double> &solution_ref, ConvergenceTable &  convergence_table_L2,  
                                                    const unsigned int fe_degree, std::string & convergence_parameter,const int convergence_resolution );
template void Process_Solutions<1>::process_refsolution<double>(Vector<double> &solution, Vector<double> &solution_ref, ConvergenceTable &  convergence_table_L2,  
                                                    const unsigned int fe_degree, std::string & convergence_parameter,const double convergence_resolution );
template void Process_Solutions<2>::process_refsolution<double>(Vector<double> &solution, Vector<double> &solution_ref, ConvergenceTable &  convergence_table_L2,  
                                                    const unsigned int fe_degree, std::string & convergence_parameter,const double convergence_resolution );
///
template void Process_Solutions<1>::process_refsolution<int>(Vector<double> &solution,Vector<double> &solution_ref,  Vector<double> &solution_prime, Vector<double> &solution_ref_prime, 
                                                    ConvergenceTable &  convergence_table_L2, ConvergenceTable &  convergence_table_H1, ConvergenceTable &  convergence_table_total,  
                                                     const unsigned int fe_degree, std::string & convergence_parameter,const int convergence_resolution );

template void Process_Solutions<2>::process_refsolution<int>(Vector<double> &solution,Vector<double> &solution_ref,  Vector<double> &solution_prime, Vector<double> &solution_ref_prime, 
                                                    ConvergenceTable &  convergence_table_L2, ConvergenceTable &  convergence_table_H1, ConvergenceTable &  convergence_table_total,  
                                                     const unsigned int fe_degree, std::string & convergence_parameter,const int convergence_resolution );
template void Process_Solutions<1>::process_refsolution<double>(Vector<double> &solution,Vector<double> &solution_ref,  Vector<double> &solution_prime, Vector<double> &solution_ref_prime, 
                                                    ConvergenceTable &  convergence_table_L2, ConvergenceTable &  convergence_table_H1, ConvergenceTable &  convergence_table_total,  
                                                     const unsigned int fe_degree, std::string & convergence_parameter,const double convergence_resolution );

template void Process_Solutions<2>::process_refsolution<double>(Vector<double> &solution,Vector<double> &solution_ref,  Vector<double> &solution_prime, Vector<double> &solution_ref_prime, 
                                                    ConvergenceTable &  convergence_table_L2, ConvergenceTable &  convergence_table_H1, ConvergenceTable &  convergence_table_total,  
                                                     const unsigned int fe_degree, std::string & convergence_parameter,const double convergence_resolution );
///
template void Process_Solutions<1>::process_exactsolution<int>(Vector<double> &solution, Solution<1> & exact_solution, ConvergenceTable &  convergence_table_L2,  
                                                    const unsigned int fe_degree, std::string & convergence_parameter,const int convergence_resolution )     ;                                                
template void Process_Solutions<2>::process_exactsolution<int>(Vector<double> &solution, Solution<2> & exact_solution, ConvergenceTable &  convergence_table_L2,  
                                                    const unsigned int fe_degree, std::string & convergence_parameter,const int convergence_resolution ) ;                                                    
template void Process_Solutions<1>::process_exactsolution<double>(Vector<double> &solution, Solution<1> & exact_solution, ConvergenceTable &  convergence_table_L2,  
                                                    const unsigned int fe_degree, std::string & convergence_parameter,const double convergence_resolution )    ;                                                 
template void Process_Solutions<2>::process_exactsolution<double>(Vector<double> &solution, Solution<2> & exact_solution, ConvergenceTable &  convergence_table_L2,  
                                                    const unsigned int fe_degree, std::string & convergence_parameter,const double convergence_resolution )  ;                                                   
///
template void Process_Solutions<1>::process_exactsolution<int>(Vector<double> &solution, Solution<1> & exact_solution,  Vector<double> &solution_prime, Solution_prime<1> & exact_solution_prime, 
                                                    ConvergenceTable &  convergence_table_L2, ConvergenceTable &  convergence_table_H1, ConvergenceTable &  convergence_table_total,  
                                                     const unsigned int fe_degree, std::string & convergence_parameter,const int convergence_resolution ) ;    
template void Process_Solutions<2>::process_exactsolution<int>(Vector<double> &solution, Solution<2> & exact_solution,  Vector<double> &solution_prime, Solution_prime<2> & exact_solution_prime, 
                                                    ConvergenceTable &  convergence_table_L2, ConvergenceTable &  convergence_table_H1, ConvergenceTable &  convergence_table_total,  
                                                     const unsigned int fe_degree, std::string & convergence_parameter,const int convergence_resolution ) ;    
template void Process_Solutions<1>::process_exactsolution<double>(Vector<double> &solution, Solution<1> & exact_solution,  Vector<double> &solution_prime, Solution_prime<1> & exact_solution_prime, 
                                                    ConvergenceTable &  convergence_table_L2, ConvergenceTable &  convergence_table_H1, ConvergenceTable &  convergence_table_total,  
                                                     const unsigned int fe_degree, std::string & convergence_parameter,const double convergence_resolution ) ;    
template void Process_Solutions<2>::process_exactsolution<double>(Vector<double> &solution, Solution<2> & exact_solution,  Vector<double> &solution_prime, Solution_prime<2> & exact_solution_prime, 
                                                    ConvergenceTable &  convergence_table_L2, ConvergenceTable &  convergence_table_H1, ConvergenceTable &  convergence_table_total,  
                                                     const unsigned int fe_degree, std::string & convergence_parameter,const double convergence_resolution ) ;    
                                                                                                  
///                                                    

}
}