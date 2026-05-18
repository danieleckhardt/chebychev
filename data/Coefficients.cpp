#include "Coefficients.h"
#include <cmath>
  
namespace MavesS // Multi-Waves-Scale
{
namespace Data
{
using namespace dealii;

    template <int dim>
    Tensor<2,dim> Coefficient_a<dim>::value( const Point<dim> &x ) const
    {
        Tensor<2,dim> return_tensor;
        if( dim == 1)
        {   if (example == -1) // Testcase 1D
                return_tensor[0][0] = 1./(2 - std::cos(2*pi*x[0]/epsilon));
            if (example == 10 || example == 12 ) // Periodic case
                return_tensor[0][0] =  1./(2 - std::cos(2*pi*x[0]/epsilon));
            if (example == 13) // Heterogeneous Medium
            {
                if( (x[0]>0 && x[0]<0.2) || (x[0]>0.3 && x[0]<0.5)  ) //(x[0]>0 && x[0]<0.5 || x[0]<1.5 && x[0]>1 ||  x[0]<2.5 && x[0]>2  || x[0]<3.5 && x[0]>3  || x[0]<4.5 && x[0]>4)
                    return_tensor[0][0] =  2. + std::sqrt(2) + 0.5*sin(2*pi*x[0]/epsilon);
                else
                    return_tensor[0][0] =  std::sqrt(2) + 0.5*sin(2*pi*x[0]/epsilon);
            }
        }
        else if(dim == 2){ 
                return_tensor[0][0] = 0.3*(1.1  + 0.5*(std::sin(2*pi*x[0]) + sin(2*pi*x[0]/epsilon)));
                return_tensor[0][1] = 0.0;
                return_tensor[1][0] = 0.0;
                return_tensor[1][1] = 0.3*(1.1  + 0.5*(std::sin(2*pi*x[0]) + sin(2*pi*x[0]/epsilon)));  
        }
        return return_tensor;
    }

    template <int dim>
    Tensor<2,dim> Coefficient_a_matrix_homogenized<dim>::value( const Point<dim> &x ) const
    {
        Tensor<2,dim> return_tensor;
        if( dim == 1)
        {
            std::cout << " Not Implemented" << std::endl;
            return Tensor<2,dim>();
        }
        else if (dim == 2)
        {
            return_tensor[0][0] = 0.3*sqrt(pow((1.1+0.5*sin(2*pi*x[0])),2)-pow(0.5,2));
            return_tensor[0][1] = 0.0;
            return_tensor[1][0] = 0.0;
            return_tensor[1][1] = 0.3*(1.1+0.5*sin(2*pi*x[0]));
            return return_tensor;
        }
    }
    template <int dim>
    double Coefficient_beta<dim>::value()
    { 
        return 0.01;
    }

    template <int dim>
    Tensor<1,dim> Coefficient_beta<dim>::value_vec(const Point<dim> &x)
    { 
        Tensor<1,dim> beta;

        if( dim == 1)
        {
            beta[0] = 1.;
        }
        else if (dim == 2)
        {
            beta[0] = -x[0];
            beta[1] = 0.0;
        }
        return beta;
    }
     
template class Coefficient_a<1>;
template class Coefficient_a<2>;
template class Coefficient_a_matrix_homogenized<1>;  
template class Coefficient_a_matrix_homogenized<2>; 
template class Coefficient_beta<1>;
template class Coefficient_beta<2>; 
}// namespace Data
}// namespace MavesS