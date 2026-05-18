#include "Output.h"

namespace MavesS {
template<int dim, typename Matrixtype, typename Vectortype>
void Output<dim, Matrixtype, Vectortype>::output_matrix_spy(Matrixtype &matrix, std::string &filename) {
  filename += "_spy.vtk";
  std::ofstream out((filename).c_str());

  for (unsigned int row = 0; row < matrix.n(); ++row) {
        typename Matrixtype::const_iterator it = matrix.begin(row);
        typename Matrixtype::const_iterator end = matrix.end(row);
        for (; it != end; ++it)
          if (fabs((*it).value()) > 10e-17)
                out << row << " -" << (*it).column() << "\t" << std::endl;
  }

  out.close();
}

template<int dim, typename Matrixtype, typename Vectortype>
void Output<dim, Matrixtype, Vectortype>::output_matrix(Matrixtype &matrix, std::string &filename) {
  filename += ".vtk";
   std::ofstream out((filename).c_str());

  for (unsigned int row = 0; row < matrix.n(); ++row) {
        typename Matrixtype::const_iterator it = matrix.begin(row);
        typename Matrixtype::const_iterator end = matrix.end(row);
        for (; it != end; ++it)
          if (fabs((*it).value()) > 10e-17)
                out << row << " -" << (*it).column() << "\t" << (*it).value() << std::endl;
  }

  out.close();
}

template<int dim, typename Matrixtype, typename Vectortype>
void Output<dim, Matrixtype, Vectortype>::output_matrix_matlab(Matrixtype &matrix, std::stringstream &filename) {
  filename << "_matlab.vtk";
  std::ofstream out((filename.str()).c_str());

  for (unsigned int row = 0; row < matrix.n(); ++row) {
        typename Matrixtype::const_iterator it = matrix.begin(row);
        typename Matrixtype::const_iterator end = matrix.end(row);
        for (; it != end; ++it)
          if (fabs((*it).value()) > 10e-17)
                out << row + 1 << "\t" << (*it).column() + 1 << "\t" << (*it).value() << std::endl;
  }

  out.close();
}
template<int dim, typename Matrixtype, typename Vectortype>
void Output<dim, Matrixtype, Vectortype>::output_matrix_txt(Matrixtype &matrix, std::string filename) {
  filename += ".txt";
  std::ofstream out((filename).c_str());

  for (unsigned int row = 0; row < matrix.n(); ++row) {
        typename Matrixtype::const_iterator it = matrix.begin(row);
        typename Matrixtype::const_iterator end = matrix.end(row);
        for (; it != end; ++it)
          if (fabs((*it).value()) > 10e-17)
                out << row  << "\t" << (*it).column()  << "\t" << (*it).value() << std::endl;
  }

  out.close();
}

template<int dim, typename Matrixtype, typename Vectortype>
void Output<dim, Matrixtype, Vectortype>::output_vector(Vectortype vector, std::string &filename )
{
  filename += ".vtk";
   std::ofstream out((filename).c_str());
  int cout = 0;
        typename Vectortype::const_iterator it = vector.begin();
        typename Vectortype::const_iterator end = vector.end();
        for (; it != end; ++it){
          out << cout  << "\t" << (*it) << std::endl;
          cout += 1;
        }
  

  out.close();
}

template<int dim, typename Matrixtype, typename Vectortype>
void Output<dim, Matrixtype, Vectortype>::output_results(DataOut<dim> data_out, std::string coefficient_type, Vector<double> const solution_macro_u,
                                                                   double const time, unsigned int const timestep_number, std::string current_Path, int epsilon ) const  
{
  std::string filename;
  if(coefficient_type == "origin") 
  {
    data_out.add_data_vector(solution_macro_u, "U_origin");
    data_out.build_patches();
    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));
    filename =
    current_Path + "/output/solution/"+ Utilities::int_to_string(dim) + "D/" +"solution" + "eps" + Utilities::int_to_string(epsilon, 3) + "-" + Utilities::int_to_string(timestep_number, 3) + ".vtk";
  }
  if(coefficient_type == "exact_hom")
  {
    data_out.add_data_vector(solution_macro_u, "U_exact_hom");
    data_out.build_patches();
    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));
    filename =
    current_Path + "/output/solution/"+ Utilities::int_to_string(dim) + "D/" +"solution_hom-";
    filename +=  Utilities::int_to_string(timestep_number, 3) + ".vtk";
  }
  if(coefficient_type == "hmm_hom")
  {
    data_out.add_data_vector(solution_macro_u, "U_hmm_hom");
    data_out.build_patches();
    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));
    filename =
    current_Path + "/output/solution/"+ Utilities::int_to_string(dim) + "D/" +"solution_hmm" + "eps" + Utilities::int_to_string(epsilon, 3) + "-" + Utilities::int_to_string(timestep_number, 3) + ".vtk";
  }
  std::ofstream output(filename);
  data_out.write_vtk(output);

} 

template class Output<1, SparseMatrix<double>>;
template class Output<1, FullMatrix<std::complex<double>>>;
template class Output<2, FullMatrix<std::complex<double>>>;
template class Output<1, FullMatrix<double>>;
template class Output<2, FullMatrix<double>>;
template class Output<2, SparseMatrix<double>>;

}

