/*
 * Output.h
 *
 *  Created on: 05.12.2016
 *      Author: koehler, carle
 *      Further developed by eckhardt
 */

#ifndef TOOLS_OUTPUT_H_
#define TOOLS_OUTPUT_H_

#include <iostream>
#include <fstream>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/numerics/data_out.h> // output .vtk

using namespace dealii;
namespace MavesS {

/**
 * \brief This class contains routines to output the assembled matrices.
 */
template<int dim, typename Matrixtype , typename Vectortype = Vector<double>>
class Output {
 public:
  /**
   * \brief Outputs the sparsity pattern of a matrix.
   */
  void output_matrix_spy(Matrixtype &matrix, std::string &filename);
  /**
   * \brief Outputs matrix in vtk format.
   */
  void output_matrix(Matrixtype &matrix, std::string &filename);
  /**
   * \brief Outputs the matrix in a form that can be importet into MatLab.
   */
  void output_matrix_matlab(Matrixtype &matrix, std::stringstream &filename);

  void output_matrix_txt(Matrixtype &matrix, std::string filename);

  void output_vector(Vectortype vector, std::string &filename );

  void output_results(DataOut<dim> data_out, std::string coefficient_type, Vector<double> const solution_macro_u,
                        double const time, unsigned int const timestep_number,std::string current_Path, int epsilon = 0) const;  

  };
} // \namespace MavesS

#endif /* TOOLS_OUTPUTMATRIX_H_ */