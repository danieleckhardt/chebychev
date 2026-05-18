#include "ExactSolution.h"
  
namespace MavesS // Multi-Waves-Scale
{
namespace Data
{
using namespace dealii;

  template <int dim>
  double Solution<dim>::value(const Point<dim> &x, const unsigned int) const
  {
    const double t = this->get_time();
    // Test case
    if(example == -1) 
      return exp(3.1415926535897931*t)*sin(3.1415926535897931*pow(x[0], 2));
    // Periodic case 
    if(example == 10)
      return exp(3.1415926535897931*t)*sin(3.1415926535897931*pow(x[0], 2));
    ///
    if(example == 12)
      return exp(3.1415926535897931*t)*sin(3.1415926535897931*pow(x[0], 2));
    ///
    else if(example == 20)
      return x[0]*x[1]*(1 - x[0])*(1 - x[1])*sin(1.5707963267948966*t);
    ///
    else if(example == 21)
      return x[0]*x[1]*(1 - x[0])*(1 - x[1])*exp(3.1415926535897931*t);
    ///
    else if(example == 22)
      return exp(3.1415926535897931*t)*sin(3.1415926535897931*pow(x[0], 2))*sin(3.1415926535897931*pow(x[1], 2));
    ///  
    else if(example == 23)
      return exp(3.1415926535897931*t)*sin(3.1415926535897931*pow(x[0], 2))*sin(3.1415926535897931*pow(x[1], 2));
    ///  
    else
    {
      Assert(false, ExcNotImplemented());
      return 0;
    }
  }
  // First Time Derivartive
  template <int dim>
  double Solution_prime<dim>::value(const Point<dim> &x, const unsigned int) const
  {
    const double t = this->get_time();
    if(example == -1)
      return 3.1415926535897931*exp(3.1415926535897931*t)*sin(3.1415926535897931*pow(x[0], 2));
    ///
    else if(example == 10)
      return 3.1415926535897931*exp(3.1415926535897931*t)*sin(3.1415926535897931*pow(x[0], 2));   
    ///  
    else if(example == 20)
      return 1.5707963267948966*x[0]*x[1]*(1 - x[0])*(1 - x[1])*cos(1.5707963267948966*t);
    ///  
    else if(example == 21)
      return 3.1415926535897931*x[0]*x[1]*(1 - x[0])*(1 - x[1])*exp(3.1415926535897931*t);
    ///
    else if(example == 22)
      return 3.1415926535897931*exp(3.1415926535897931*t)*sin(3.1415926535897931*pow(x[0], 2))*sin(3.1415926535897931*pow(x[1], 2));
    ///
    else if(example == 23)
      return 3.1415926535897931*exp(3.1415926535897931*t)*sin(3.1415926535897931*pow(x[0], 2))*sin(3.1415926535897931*pow(x[1], 2));
    ///
    else
    {
      Assert(false, ExcNotImplemented());
      return 0;
    }
  }
  // Gradient
  template <int dim>
  Tensor<1, dim> Solution<dim>::gradient(const Point<dim> &x,const unsigned int) const
  { 
    const double t = this->get_time();
    Tensor<1,dim> return_value;
    if(example == -1)
      return_value[0] = 6.2831853071795862*x[0]*exp(3.1415926535897931*t)*cos(3.1415926535897931*pow(x[0], 2)); 
    ///
    else if( example == 10)
      return_value[0] = 6.2831853071795862*x[0]*exp(3.1415926535897931*t)*cos(3.1415926535897931*pow(x[0], 2)); 
    ///
    else if(example == 20)
    {
      return_value[0] = -x[0]*x[1]*(1 - x[1])*sin(1.5707963267948966*t) + x[1]*(1 - x[0])*(1 - x[1])*sin(1.5707963267948966*t);
      return_value[1] = -x[0]*x[1]*(1 - x[0])*sin(1.5707963267948966*t) + x[0]*(1 - x[0])*(1 - x[1])*sin(1.5707963267948966*t);
    }
    ///
    else if(example == 21)
    {
      return_value[0] = -x[0]*x[1]*(1 - x[1])*exp(3.1415926535897931*t) + x[1]*(1 - x[0])*(1 - x[1])*exp(3.1415926535897931*t);
      return_value[1] = -x[0]*x[1]*(1 - x[0])*exp(3.1415926535897931*t) + x[0]*(1 - x[0])*(1 - x[1])*exp(3.1415926535897931*t);
    }
    ///
    else if(example == 22)
    {
      return_value[0] = 6.2831853071795862*x[0]*exp(3.1415926535897931*t)*sin(3.1415926535897931*pow(x[1], 2))*cos(3.1415926535897931*pow(x[0], 2));
      return_value[1] = 6.2831853071795862*x[1]*exp(3.1415926535897931*t)*sin(3.1415926535897931*pow(x[0], 2))*cos(3.1415926535897931*pow(x[1], 2));
    }
    ///
    else if(example == 23)
    {
      return_value[0] = 6.2831853071795862*x[0]*exp(3.1415926535897931*t)*sin(3.1415926535897931*pow(x[1], 2))*cos(3.1415926535897931*pow(x[0], 2));
      return_value[1] = 6.2831853071795862*x[1]*exp(3.1415926535897931*t)*sin(3.1415926535897931*pow(x[0], 2))*cos(3.1415926535897931*pow(x[1], 2));
    }
    ///
    else
      Assert(false, ExcNotImplemented());

    return return_value;
  }
 
template class Solution<1>;
template class Solution<2>;
template class Solution_prime<1>;
template class Solution_prime<2>;
}// namespace Data
}// namespace MavesS
