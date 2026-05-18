#include "MicroProblem.h"

using namespace dealii;

namespace MavesS // Multi-Waves-Scale
{
using namespace Data;
namespace Microproblem
{
  template <int dim>
  GeneratingHomCoeff<dim>::GeneratingHomCoeff(double epsilon,double delta,bool periodic,int example_config, unsigned int level_micro,unsigned int order_micro,DoFHandler<dim> &dof_handler_macro,
   FE_Q<dim> &fe_macro,  AffineConstraints<double>  &constraints_macro, TimerOutput &  computing_timer)
      :
       computing_timer(computing_timer)
      ,epsilon(epsilon)
      ,delta(delta)
      ,periodic(periodic)
      ,example_config(example_config)
      ,level_micro(level_micro)
      ,fe_micro(order_micro) // quadratic
      ,dof_handler_micro(triangulation_micro)
      ,fe_macro(fe_macro)
      ,constraints_macro(constraints_macro)
      ,dof_handler_macro(dof_handler_macro)
      ,mapping()
      ,quadrature(fe_macro.tensor_degree() + 1)
      ,coefficient(epsilon, example_config)
      {}
            

template <int dim>
void GeneratingHomCoeff<dim>::setup_system_and_triang_micro(){
    TimerOutput::Scope t(computing_timer, "setup_system_and_triang_micro");

    // Clear old triangulation and generate a hypercube microstructure
    triangulation_micro.clear();
    GridGenerator::hyper_cube(triangulation_micro, -delta / 2., delta / 2., periodic);
    triangulation_micro.refine_global(level_micro);

    // Distribute degrees of freedom
    dof_handler_micro.distribute_dofs(fe_micro);

    // Initialize constraints
    constraints_micro.clear();

    // Apply periodic boundary constraints if enabled
    if (periodic)
    {
        for (unsigned int i = 0; i < dim; ++i)
        {
            std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>> periodic_faces;
            GridTools::collect_periodic_faces(dof_handler_micro, 2 * i, 2 * i + 1, i, periodic_faces);
            DoFTools::make_periodicity_constraints<dim, dim>(periodic_faces, constraints_micro);
        }
        constraints_micro.close();
    }

    // Create sparsity pattern with or without constraints
    DynamicSparsityPattern dsp(dof_handler_micro.n_dofs());

    if (periodic)
        DoFTools::make_sparsity_pattern(dof_handler_micro, dsp, constraints_micro);
    else
        DoFTools::make_sparsity_pattern(dof_handler_micro, dsp);

    sparsity_pattern_micro.copy_from(dsp);
}

  template <int dim>
  void GeneratingHomCoeff<dim>::assemble_solve_micro(Point<dim> p, Tensor<1,dim> gradphi, Vector<double>& solution ) { 
    //TimerOutput::Scope t(computing_timer, "assemble_solve_micro");
    SparseMatrix<double>       system_matrix_micro;
    Vector<double>             system_rhs_micro;

    system_matrix_micro.reinit(sparsity_pattern_micro);
    system_rhs_micro.reinit(dof_handler_micro.n_dofs());
    system_matrix_micro = 0;
    system_rhs_micro = 0;

    QGauss<dim> quadrature_formula(fe_micro.tensor_degree()); 
    FEValues<dim> fe_values (fe_micro, quadrature_formula, update_values |  update_quadrature_points | update_gradients | update_JxW_values);

    const unsigned int dofs_per_cell = fe_micro.dofs_per_cell; 

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>    cell_rhs(dofs_per_cell); 

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    for (const auto &cell : dof_handler_micro.active_cell_iterators())
    {

      fe_values.reinit(cell);

      cell_matrix = 0;
      cell_rhs    = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        const Tensor<2,dim> current_coefficient = coefficient.value(fe_values.quadrature_point(q_index) + p); 

        for (const unsigned int i : fe_values.dof_indices())
        {
            for (const unsigned int j : fe_values.dof_indices()){
                cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                   current_coefficient *              // a(x_K + quadrature_point)
                  fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                  fe_values.JxW(q_index));           // dx
            }

            cell_rhs(i) += -(fe_values.shape_grad(i, q_index) * // -grad phi_i(x_q) 
                            current_coefficient *               // a(x_q)
                            gradphi *                           // grad Phi_macro(x_K)
                            fe_values.JxW(q_index));           // dx
  
        }
      }
        cell->get_dof_indices(local_dof_indices);

        for (const unsigned int i : fe_values.dof_indices()){
          for (const unsigned int j : fe_values.dof_indices()){
            system_matrix_micro.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));
          }
          system_rhs_micro(local_dof_indices[i]) += cell_rhs(i);
        }
    }
    if (periodic)
    {
            constraints_micro.condense(system_matrix_micro);
            constraints_micro.condense(system_rhs_micro);
    }
    else{
    std::map<types::global_dof_index, double> boundary_values0;
    std::map<types::global_dof_index, double> boundary_values1;

    VectorTools::interpolate_boundary_values(dof_handler_micro,
                                            0,
                                            Functions::ZeroFunction<dim>(),
                                            boundary_values0);
    MatrixTools::apply_boundary_values(boundary_values0,
                                      system_matrix_micro,
                                      solution,
                                      system_rhs_micro);

   //for 1D Problems the right-handside boundary has the index 1.
    if(dim == 1)
    {
    VectorTools::interpolate_boundary_values(dof_handler_micro,
                                                  1,
                                                  Functions::ZeroFunction<dim>(),
                                                  boundary_values1);

    MatrixTools::apply_boundary_values(boundary_values1,
                                            system_matrix_micro,
                                            solution,
                                            system_rhs_micro);
    }
    }
    solve_micro(system_rhs_micro, system_matrix_micro,solution); 
    if(periodic)
      constraints_micro.distribute(solution);

  }

  template <int dim>
  void GeneratingHomCoeff<dim>::assemble_B_H(SparseMatrix<double> &homogeneous_matrix){
    TimerOutput::Scope t(computing_timer, "assemble_B_H");

    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
 
    const auto cell_worker = [&](const Iterator &  cell,
                                 ScratchDataMacro<dim> &scratch_data_macro,
                                 CopyDataMacro &      copy_data) 
    {
      const unsigned int n_dofs =
        scratch_data_macro.fe_values_macro.get_fe().n_dofs_per_cell();
      copy_data.reinit(cell, n_dofs);
      scratch_data_macro.fe_values_macro.reinit(cell);

      //const QGauss<dim> quadrature_formula = scratch_data.fe_values_macro.get_quadrature();
      const unsigned int q_points = quadrature.size();

      const FEValues<dim> &fe_val_macro = scratch_data_macro.fe_values_macro;
      const std::vector<double> &JxW  = fe_val_macro.get_JxW_values();

      Vector<double>                   solution_micro(dof_handler_micro.n_dofs());  //Vector of solutions of the linear micro-problem 
                                                                                          //on the given QP with coefficient a_eps 
      std::vector<Tensor<1,dim>>       solution_micro_grad(q_points);  // Vector of the gradients of the micro solution at quadrature points
      FEValues<dim> fe_values_micro (fe_micro, quadrature, update_values | update_quadrature_points | update_gradients | update_JxW_values);

      //  Auxiliary variablen
      Point<dim>                       q_point_micro, q_point_macro;
      double                           dx_micro;
      Tensor<1,dim>                    grad_q_i; 

      for (unsigned int q_mac=0; q_mac<q_points; ++q_mac){
        q_point_macro = fe_val_macro.quadrature_point(q_mac); 

        for (unsigned int i =0; i<fe_macro.dofs_per_cell; ++i){ 
          grad_q_i = fe_val_macro.shape_grad(i,q_mac); //Gradient of the i-th macro basis function at the macro QP
          // Assembly of the homogeneous matrix 
          assemble_solve_micro(q_point_macro, grad_q_i,solution_micro) ; 

          //Loop over the elemets of the micro triangulation
          for (const auto &cell_micro : dof_handler_micro.active_cell_iterators()){
            fe_values_micro.reinit(cell_micro);
            fe_values_micro.get_function_gradients(solution_micro, solution_micro_grad);
            //Loop over the micro quadrature points 
            for (unsigned int q_mic=0; q_mic<q_points; ++q_mic){
              dx_micro = fe_values_micro.JxW(q_mic);  //Determinante für die Transformation in die Micro-Referenzzelle
              q_point_micro = fe_values_micro.quadrature_point(q_mic)+q_point_macro;; //Weights for the macro quadrature point
              const Tensor<2,dim> current_coefficient = coefficient.value(q_point_micro); // ÄNDERUNG!! Macro falsch 
            
              for (unsigned int j =0; j<fe_macro.dofs_per_cell; ++j){ 
                copy_data.cell_matrix(i,j)+= 1/std::pow(this->delta,dim)*(fe_val_macro.shape_grad(j,q_mac)*current_coefficient
                                                        *(solution_micro_grad[q_mic] + grad_q_i)*dx_micro)*JxW[q_mac];
              } 
            }
          }
        }
      }
    };
    // const AffineConstraints<double> constraints;
  
    const auto copier = [&](const CopyDataMacro &c) {
        constraints_macro.distribute_local_to_global(c.cell_matrix, c.local_dof_indices, homogeneous_matrix);
    };
    ScratchDataMacro<dim> scratch_data_macro(mapping, fe_macro, quadrature);
    CopyDataMacro         copy_data;
    MeshWorker::mesh_loop(dof_handler_macro.begin_active(),
                          dof_handler_macro.end(),
                          cell_worker,
                          copier,
                          scratch_data_macro,
                          copy_data,
                          MeshWorker::assemble_own_cells);
  /*                         ,
                          nullptr,
                          nullptr,
                          4,
                          8
                          ); */

  }

  template <int dim>
  void GeneratingHomCoeff<dim>::assemble_damping_with_cellproblem(SparseMatrix<double> &damp_matrix){
    TimerOutput::Scope t(computing_timer, "assemble_damping_with_cellproblem");

    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
 
    const auto cell_worker = [&](const Iterator &  cell,
                                 ScratchDataMacro<dim> &scratch_data_macro,
                                 CopyDataMacro &      copy_data) 
    {
      const unsigned int n_dofs =
        scratch_data_macro.fe_values_macro.get_fe().n_dofs_per_cell();
      copy_data.reinit(cell, n_dofs);
      scratch_data_macro.fe_values_macro.reinit(cell);

      //const QGauss<dim> quadrature_formula = scratch_data.fe_values_macro.get_quadrature();
      const unsigned int q_points = quadrature.size();

      const FEValues<dim> &fe_val_macro = scratch_data_macro.fe_values_macro;
      const std::vector<double> &JxW  = fe_val_macro.get_JxW_values();

      Vector<double>                   solution_micro(dof_handler_micro.n_dofs());  //Vector of solutions of the linear micro-problem 
                                                                                          //on the given QP with coefficient a_eps 
      std::vector<Tensor<1,dim>>       solution_micro_grad(q_points);  // Vector of the gradients of the micro solution at quadrature points
      FEValues<dim> fe_values_micro (fe_micro, quadrature, update_values | update_quadrature_points | update_gradients | update_JxW_values);

      //  Auxiliary variablen
      Point<dim>                       q_point_micro, q_point_macro;
      double                           dx_micro;
      Tensor<1,dim>                    grad_q_i; 
      double test;

      for (unsigned int q_mac=0; q_mac<q_points; ++q_mac){
        q_point_macro = fe_val_macro.quadrature_point(q_mac); 

        for (unsigned int i =0; i<fe_macro.dofs_per_cell; ++i){ 
          grad_q_i = fe_val_macro.shape_grad(i,q_mac); //Gradient of the i-th macro basis function at the macro QP
          // Assembly of the homogeneous matrix 
          assemble_solve_micro(q_point_macro, grad_q_i,solution_micro) ; 

          //Loop over the elemets of the micro triangulation
          for (const auto &cell_micro : dof_handler_micro.active_cell_iterators()){
            fe_values_micro.reinit(cell_micro);
            fe_values_micro.get_function_gradients(solution_micro, solution_micro_grad);
            //Loop over the micro quadrature points 
            for (unsigned int q_mic=0; q_mic<q_points; ++q_mic){
              dx_micro = fe_values_micro.JxW(q_mic);  //Determinante für die Transformation in die Micro-Referenzzelle
              q_point_micro = fe_values_micro.quadrature_point(q_mic)+q_point_macro;; //Weights for the macro quadrature point
            
              for (unsigned int j =0; j<fe_macro.dofs_per_cell; ++j){ 
                test = 1/std::pow(this->delta,dim) *fe_val_macro.shape_grad(j,q_mac)*(grad_q_i + solution_micro_grad[q_mic])*dx_micro*JxW[q_mac];
                std::cout<< test<< std::endl;
                copy_data.cell_matrix(i,j)+= 1/std::pow(this->delta,dim)*(fe_val_macro.shape_grad(j,q_mac) *(solution_micro_grad[q_mic] + grad_q_i)*dx_micro)*JxW[q_mac];
              } 
            }
          }
        }
      }
    };
    // const AffineConstraints<double> constraints;
  
    const auto copier = [&](const CopyDataMacro &c) {
        constraints_macro.distribute_local_to_global(c.cell_matrix, c.local_dof_indices, damp_matrix);
    };
    ScratchDataMacro<dim> scratch_data_macro(mapping, fe_macro, quadrature);
    CopyDataMacro         copy_data;
    MeshWorker::mesh_loop(dof_handler_macro.begin_active(),
                          dof_handler_macro.end(),
                          cell_worker,
                          copier,
                          scratch_data_macro,
                          copy_data,
                          MeshWorker::assemble_own_cells);
  /*                         ,
                          nullptr,
                          nullptr,
                          4,
                          8
                          ); */

  }

  template <int dim>
  void GeneratingHomCoeff<dim>::assemble_longtime(SparseMatrix<double> &effdamp_matrix){
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
    const auto cell_worker = [&](const Iterator &  cell,
                                 ScratchDataMacro<dim> &scratch_data_macro,
                                 CopyDataMacro &      copy_data) 
    {
      const unsigned int n_dofs =
        scratch_data_macro.fe_values_macro.get_fe().n_dofs_per_cell();
      copy_data.reinit(cell, n_dofs);
      scratch_data_macro.fe_values_macro.reinit(cell);

      //const QGauss<dim> quadrature_formula = scratch_data.fe_values_macro.get_quadrature();
      const unsigned int q_points = quadrature.size();

      const FEValues<dim> &fe_val_macro = scratch_data_macro.fe_values_macro;
      const std::vector<double> &JxW  = fe_val_macro.get_JxW_values();

      Vector<double>                   solution_micro_i(dof_handler_micro.n_dofs());  //Vector of solutions of the linear micro-problem 
                                                                                          //on the given QP with coefficient a_eps 

      Vector<double>                   solution_micro_j(dof_handler_micro.n_dofs());  //Vector of solutions of the linear micro-problem 
                                                                                          //on the given QP with coefficient a_eps 
      std::vector<Tensor<1,dim>>       solution_micro_grad_i(q_points);  // Vector of the gradients of the micro solution at quadrature points
      std::vector<Tensor<1,dim>>       solution_micro_grad_j(q_points);  // Vector of the gradients of the micro solution at quadrature points

      FEValues<dim> fe_values_micro (fe_micro, quadrature, update_values | update_quadrature_points | update_gradients | update_JxW_values);

      //  Auxiliary variablen
      Point<dim>                       q_point_micro, q_point_macro;
      double                           dx_micro;
      Tensor<1,dim>                    grad_q_i; 
      Tensor<1,dim>                    grad_q_j; 

      for (unsigned int q_mac=0; q_mac<q_points; ++q_mac){
        q_point_macro = fe_val_macro.quadrature_point(q_mac); 

        for (unsigned int i =0; i<fe_macro.dofs_per_cell; ++i){ 
           grad_q_i = fe_val_macro.shape_grad(i,q_mac); //Gradient of the i-th macro basis function at the macro QP
          // Assembly of the homogeneous matrix 
          assemble_solve_micro(q_point_macro, grad_q_i,solution_micro_i) ; 
          //std::cout<< solution_micro_i.l2_norm();
          for (unsigned int j =0; j<fe_macro.dofs_per_cell; ++j){ 
            grad_q_j = fe_val_macro.shape_grad(j,q_mac); //Gradient of the i-th macro basis function at the macro QP
            // Assembly of the homogeneous matrix 
            assemble_solve_micro(q_point_macro, grad_q_j,solution_micro_j) ; 

          //Loop over the elemets of the micro triangulation
          for (const auto &cell_micro : dof_handler_micro.active_cell_iterators()){
            fe_values_micro.reinit(cell_micro);
            fe_values_micro.get_function_gradients(solution_micro_i, solution_micro_grad_i);
            fe_values_micro.get_function_gradients(solution_micro_j, solution_micro_grad_j);
            //Loop over the micro quadrature points 
            for (unsigned int q_mic=0; q_mic<q_points; ++q_mic){
              dx_micro = fe_values_micro.JxW(q_mic);  //Determinante für die Transformation in die Micro-Referenzzelle
              q_point_micro = fe_values_micro.quadrature_point(q_mic)+q_point_macro;; //Weights for the macro quadrature point
              //const Tensor<2,dim> current_coefficient = coefficient.value(q_point_micro); // ÄNDERUNG!! Macro falsch 
              copy_data.cell_matrix(i,j)+=  1/std::pow(this->delta,dim)*( solution_micro_j[q_mic]
                                                        *(solution_micro_i[q_mic])*dx_micro)*JxW[q_mac];
              } 
            }
          }
        }
      }
    };
  
    const auto copier = [&](const CopyDataMacro &c) {
        constraints_macro.distribute_local_to_global(c.cell_matrix, c.local_dof_indices, effdamp_matrix);
    };
    ScratchDataMacro<dim> scratch_data_macro(mapping, fe_macro, quadrature);
    CopyDataMacro         copy_data;
    MeshWorker::mesh_loop(dof_handler_macro.begin_active(),
                          dof_handler_macro.end(),
                          cell_worker,
                          copier,
                          scratch_data_macro,
                          copy_data,
                          MeshWorker::assemble_own_cells);

  }

  template <int dim>
  void GeneratingHomCoeff<dim>::assemble_B_H_symmetric(SparseMatrix<double> &homogeneous_matrix){
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
    const auto cell_worker = [&](const Iterator &  cell,
                                 ScratchDataMacro<dim> &scratch_data_macro,
                                 CopyDataMacro &      copy_data) 
    {
      const unsigned int n_dofs =
        scratch_data_macro.fe_values_macro.get_fe().n_dofs_per_cell();
      copy_data.reinit(cell, n_dofs);
      scratch_data_macro.fe_values_macro.reinit(cell);

      //const QGauss<dim> quadrature_formula = scratch_data.fe_values_macro.get_quadrature();
      const unsigned int q_points = quadrature.size();

      const FEValues<dim> &fe_val_macro = scratch_data_macro.fe_values_macro;
      const std::vector<double> &JxW  = fe_val_macro.get_JxW_values();

      Vector<double>                   solution_micro_i(dof_handler_micro.n_dofs());  //Vector of solutions of the linear micro-problem 
                                                                                          //on the given QP with coefficient a_eps 

      Vector<double>                   solution_micro_j(dof_handler_micro.n_dofs());  //Vector of solutions of the linear micro-problem 
                                                                                          //on the given QP with coefficient a_eps 
      std::vector<Tensor<1,dim>>       solution_micro_grad_i(q_points);  // Vector of the gradients of the micro solution at quadrature points
      std::vector<Tensor<1,dim>>       solution_micro_grad_j(q_points);  // Vector of the gradients of the micro solution at quadrature points

      FEValues<dim> fe_values_micro (fe_micro, quadrature, update_values | update_quadrature_points | update_gradients | update_JxW_values);

      //  Auxiliary variablen
      Point<dim>                       q_point_micro, q_point_macro;
      double                           dx_micro;
      Tensor<1,dim>                    grad_q_i; 
      Tensor<1,dim>                    grad_q_j; 

      for (unsigned int q_mac=0; q_mac<q_points; ++q_mac){
        q_point_macro = fe_val_macro.quadrature_point(q_mac); 

        for (unsigned int i =0; i<fe_macro.dofs_per_cell; ++i){ 
           grad_q_i = fe_val_macro.shape_grad(i,q_mac); //Gradient of the i-th macro basis function at the macro QP
          // Assembly of the homogeneous matrix 
          assemble_solve_micro(q_point_macro, grad_q_i,solution_micro_i) ; 
          //std::cout<< solution_micro_i.l2_norm();
          for (unsigned int j =0; j<fe_macro.dofs_per_cell; ++j){ 
            grad_q_j = fe_val_macro.shape_grad(j,q_mac); //Gradient of the i-th macro basis function at the macro QP
            // Assembly of the homogeneous matrix 
            assemble_solve_micro(q_point_macro, grad_q_j,solution_micro_j) ; 

          //Loop over the elemets of the micro triangulation
          for (const auto &cell_micro : dof_handler_micro.active_cell_iterators()){
            fe_values_micro.reinit(cell_micro);
            fe_values_micro.get_function_gradients(solution_micro_i, solution_micro_grad_i);
            fe_values_micro.get_function_gradients(solution_micro_j, solution_micro_grad_j);
            //Loop over the micro quadrature points 
            for (unsigned int q_mic=0; q_mic<q_points; ++q_mic){
              dx_micro = fe_values_micro.JxW(q_mic);  //Determinante für die Transformation in die Micro-Referenzzelle
              q_point_micro = fe_values_micro.quadrature_point(q_mic)+q_point_macro;; //Weights for the macro quadrature point
              const Tensor<2,dim> current_coefficient = coefficient.value(q_point_micro); // ÄNDERUNG!! Macro falsch 
              copy_data.cell_matrix(i,j)+=  1/std::pow(this->delta,dim)*(solution_micro_grad_j[q_mic] + grad_q_j )*current_coefficient
                                                        *(solution_micro_grad_i[q_mic] + grad_q_i)*dx_micro*JxW[q_mac];
              } 
            }
          }
        }
      }
    };
  
    const auto copier = [&](const CopyDataMacro &c) {
        constraints_macro.distribute_local_to_global(c.cell_matrix, c.local_dof_indices, homogeneous_matrix);
    };
    ScratchDataMacro<dim> scratch_data_macro(mapping, fe_macro, quadrature);
    CopyDataMacro         copy_data;
    MeshWorker::mesh_loop(dof_handler_macro.begin_active(),
                          dof_handler_macro.end(),
                          cell_worker,
                          copier,
                          scratch_data_macro,
                          copy_data,
                          MeshWorker::assemble_own_cells);

  }
  
  template <int dim>
  void GeneratingHomCoeff<dim>::assemble_B_H_origin(SparseMatrix<double> &homogeneous_matrix){
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
 
    const auto cell_worker = [&](const Iterator &  cell,
                                 ScratchDataMacro<dim> &scratch_data_macro,
                                 CopyDataMacro &      copy_data) {
    const unsigned int n_dofs =
      scratch_data_macro.fe_values_macro.get_fe().n_dofs_per_cell();
    copy_data.reinit(cell, n_dofs);
    scratch_data_macro.fe_values_macro.reinit(cell);

    //const QGauss<dim> quadrature_formula = scratch_data.fe_values_macro.get_quadrature();
    const unsigned int q_points = quadrature.size();

    const FEValues<dim> &fe_val_macro = scratch_data_macro.fe_values_macro;
    const std::vector<double> &JxW  = fe_val_macro.get_JxW_values();

    //  Auxiliary variablen
    Point<dim>                       q_point_macro;

    for (unsigned int q_mac=0; q_mac<q_points; ++q_mac){
      q_point_macro = fe_val_macro.quadrature_point(q_mac); 
      const Tensor<2,dim> current_coefficient = coefficient.value(q_point_macro);
      for (unsigned int i =0; i<fe_macro.dofs_per_cell; ++i){ 
        //Loop over the elemets of the micro triangulation          
            for (unsigned int j =0; j<fe_macro.dofs_per_cell; ++j){ 
              copy_data.cell_matrix(i,j)+= (fe_val_macro.shape_grad(j,q_mac)*current_coefficient
                                                      *fe_val_macro.shape_grad(i,q_mac))*JxW[q_mac];
        }
      }
    }
  };
  // const AffineConstraints<double> constraints;
 
  const auto copier = [&](const CopyDataMacro &c) {
      constraints_macro.distribute_local_to_global(c.cell_matrix, c.local_dof_indices, homogeneous_matrix);
  };
  ScratchDataMacro<dim> scratch_data_macro(mapping, fe_macro, quadrature);
  CopyDataMacro         copy_data;
  MeshWorker::mesh_loop(dof_handler_macro.begin_active(),
                        dof_handler_macro.end(),
                        cell_worker,
                        copier,
                        scratch_data_macro,
                        copy_data,
                        MeshWorker::assemble_own_cells);
  }

  template <int dim>
  void GeneratingHomCoeff<dim>::assemble_B_H_homogenized(SparseMatrix<double> &homogeneous_matrix){
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
 
    const auto cell_worker = [&](const Iterator &  cell,
                                 ScratchDataMacro<dim> &scratch_data_macro,
                                 CopyDataMacro &      copy_data) {
    const unsigned int n_dofs =
      scratch_data_macro.fe_values_macro.get_fe().n_dofs_per_cell();
    copy_data.reinit(cell, n_dofs);
    scratch_data_macro.fe_values_macro.reinit(cell);

    //const QGauss<dim> quadrature_formula = scratch_data.fe_values_macro.get_quadrature();
    const unsigned int q_points = quadrature.size();

    const FEValues<dim> &fe_val_macro = scratch_data_macro.fe_values_macro;
    const std::vector<double> &JxW  = fe_val_macro.get_JxW_values();

    //  Auxiliary variablen
    Point<dim>                       q_point_macro;

    for (unsigned int q_mac=0; q_mac<q_points; ++q_mac){
      q_point_macro = fe_val_macro.quadrature_point(q_mac); 
      Tensor<2,dim> current_coefficient = coefficient_matrix_homogenized.value(q_point_macro);
      for (unsigned int i =0; i<fe_macro.dofs_per_cell; ++i){ 
        //Loop over the elemets of the micro triangulation          
            for (unsigned int j =0; j<fe_macro.dofs_per_cell; ++j){ 
              copy_data.cell_matrix(i,j)+= (fe_val_macro.shape_grad(j,q_mac)*current_coefficient
                                                      *fe_val_macro.shape_grad(i,q_mac))*JxW[q_mac];
        }
      }
    }
  };
  // const AffineConstraints<double> constraints;
 
  const auto copier = [&](const CopyDataMacro &c) {
      constraints_macro.distribute_local_to_global(c.cell_matrix, c.local_dof_indices, homogeneous_matrix);
  };
  ScratchDataMacro<dim> scratch_data_macro(mapping, fe_macro, quadrature);
  CopyDataMacro         copy_data;
  MeshWorker::mesh_loop(dof_handler_macro.begin_active(),
                        dof_handler_macro.end(),
                        cell_worker,
                        copier,
                        scratch_data_macro,
                        copy_data,
                        MeshWorker::assemble_own_cells);
  }

  template <int dim>
void GeneratingHomCoeff<dim>::assemble_damping_grad(
    SparseMatrix<double> &damping_matrix)
{
  using Iterator =
    typename DoFHandler<dim>::active_cell_iterator;

  const auto cell_worker =
    [&](const Iterator &cell,
        ScratchDataMacro<dim> &scratch_data_macro,
        CopyDataMacro &copy_data)
  {
    scratch_data_macro.fe_values_macro.reinit(cell);

    const FEValues<dim> &fe_val =
      scratch_data_macro.fe_values_macro;

    const unsigned int n_dofs =
      fe_val.get_fe().n_dofs_per_cell();

    const unsigned int q_points =
      fe_val.n_quadrature_points;

    copy_data.reinit(cell, n_dofs);

    const std::vector<double> &JxW =
      fe_val.get_JxW_values();

    for (unsigned int q = 0; q < q_points; ++q)
    {
      const Point<dim> x_q =
        fe_val.quadrature_point(q);

      // beta : Tensor<1,dim>
      const Tensor<1,dim> beta =
        coefficient_beta.value_vec(x_q);

      for (unsigned int j = 0; j < n_dofs; ++j)
      {
        const Tensor<1,dim> grad_phi_j =
          fe_val.shape_grad(j,q);

        const double beta_grad_j =
          beta * grad_phi_j;

        for (unsigned int i = 0; i < n_dofs; ++i)
        {
          copy_data.cell_matrix(i,j) +=
            beta_grad_j *
            fe_val.shape_value(i,q) *
            JxW[q];
        }
      }
    }
  };

  const auto copier =
    [&](const CopyDataMacro &c)
  {
    constraints_macro.distribute_local_to_global(
        c.cell_matrix,
        c.local_dof_indices,
        damping_matrix);
  };

  ScratchDataMacro<dim> scratch_data_macro(
      mapping, fe_macro, quadrature);
  CopyDataMacro copy_data;

  MeshWorker::mesh_loop(
      dof_handler_macro.begin_active(),
      dof_handler_macro.end(),
      cell_worker,
      copier,
      scratch_data_macro,
      copy_data,
      MeshWorker::assemble_own_cells);
}


  template <int dim>
  void GeneratingHomCoeff<dim>::solve_micro(Vector<double>& rhs, SparseMatrix<double>& matrix, Vector<double>& loesung){
    SolverControl           solver_control(4000, 1e-10); //1e-10
    SolverCG<Vector<double>> solver(solver_control);
    PreconditionJacobi<SparseMatrix<double>> preconditioner;
    // PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(matrix);
    solver.solve (matrix, loesung, rhs, preconditioner);
  }


template class GeneratingHomCoeff<1>; 
template class GeneratingHomCoeff<2>; 
}
}
