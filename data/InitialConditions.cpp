#include "InitialConditions.h"
#include <cmath>

namespace MavesS // Multi-Waves-Scale
{
namespace Data{
using namespace dealii;

 template<int dim , typename number>
  number InitialCondition<dim,number>::value(const Point<dim> &x,
                                    const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    Assert(dim < 3, ExcNotImplemented());
    ///
    if(example == -1)
    {
      switch(component_init)
      {
        case U:
        return sin(3.1415926535897931*pow(x[0], 2));
        case V:
        return 3.1415926535897931*sin(3.1415926535897931*pow(x[0], 2));
        default:
            Assert(false, ExcNotImplemented());
      }
    }
    ///
    else if(example == 10)
    {
      switch(component_init)
      {
        case U:
          return sin(3.1415926535897931*pow(x[0], 2));
        case V:
          return 3.1415926535897931*sin(3.1415926535897931*pow(x[0], 2));
        default:
            Assert(false, ExcNotImplemented());
      }
    }
    ///
    else if(example == 12)
    { 
      switch(component_init)
        {
          case U:
            return exp(-(x[0] - 0.5)*(x[0] - 0.5)*gaus); // gauss beam;
          case V:
            return 2*(x[0]- 0.5)*gaus*exp(-((x[0]- 0.5)*(x[0]- 0.5))*gaus );
          default:
            Assert(false, ExcNotImplemented());
        }
    }
    ///
    else if(example == -2)
    { 
      switch(component_init)
        {
          case U:
            return  0;
          case V:
            return  1.5707963267948966*x[0]*x[1]*(1 - x[0])*(1 - x[1]);
          default:
            Assert(false, ExcNotImplemented());
        }
    }
    ///
    ///
    else if(example == 20)
    { 
      switch(component_init)
        {
          case U:
            return  0;
          case V:
            return  1.5707963267948966*x[0]*x[1]*(1 - x[0])*(1 - x[1]);
          default:
            Assert(false, ExcNotImplemented());
        }
    }
    ///
    else if(example == 21)
    { 
      switch(component_init)
        {
          case U:
            return x[0]*x[1]*(1 - x[0])*(1 - x[1]); // gauss beam;
          case V:
            return 3.1415926535897931*x[0]*x[1]*(1 - x[0])*(1 - x[1]);
          default:
            Assert(false, ExcNotImplemented());
        }
    }
    ///
    else if(example == 22)
    { 
      switch(component_init)
        {
          case U:
            return sin(3.1415926535897931*pow(x[0], 2))*sin(3.1415926535897931*pow(x[1], 2));
          case V:
            return 3.1415926535897931*sin(3.1415926535897931*pow(x[0], 2))*sin(3.1415926535897931*pow(x[1], 2));
          default:
            Assert(false, ExcNotImplemented());
        }
    }
    ///
    else if(example == 23)
    { 
      switch(component_init)
        {
          case U:
            return sin(3.1415926535897931*pow(x[0], 2))*sin(3.1415926535897931*pow(x[1], 2));
          case V:
            return 3.1415926535897931*sin(3.1415926535897931*pow(x[0], 2))*sin(3.1415926535897931*pow(x[1], 2));
          default:
            Assert(false, ExcNotImplemented());
        }
    }
    ///
    else
      Assert(false, ExcNotImplemented());
    return 0;
  }


template class InitialCondition<1>;
template class InitialCondition<1, std::complex<double>>;
template class InitialCondition<2>;
template class InitialCondition<2, std::complex<double>>;
}// namespace Data
}// namespace MavesS