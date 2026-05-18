#ifndef RHS_H_
#define RHS_H_


#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <cmath>

namespace MavesS // Multi-Waves-Scale
{
namespace Data{
using namespace dealii;

template <int dim>
  class RightHandSide : public Function<dim>
  {
    public:
      RightHandSide(const int example)
      : Function<dim>()
      ,example(example)
      {}

      virtual double value(const Point<dim> & p,
                              const unsigned int component = 0) const override;
    private:
    const int example;
    };

}// namespace Data
}// namespace MavesS
 #endif /* RHS_H_ */