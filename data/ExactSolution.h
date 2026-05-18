#ifndef EXACTSOLUTION_H_
#define EXACTSOLUTION_H_

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

#include <cmath>

namespace MavesS // Multi-Waves-Scale
{
namespace Data
{
using namespace dealii;

  template<int dim>
  class Solution : public Function<dim> // class for the exact solution
  {
  public: 
    Solution( const int example)
        : Function<dim>()
        ,example(example)
        {}

    virtual double value(const Point<dim> & p,
                        const unsigned int component = 0) const override;
    virtual Tensor<1,dim> gradient(const Point<dim> & p,
                        const unsigned int component = 0) const override;  
    private:
      const int example;
  };

  template<int dim>
  class Solution_prime : public Function<dim> // class for the exact solution
  {
  public: 
      Solution_prime( const int example)
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
 #endif /* EXACTSOLUTION_H_ */


