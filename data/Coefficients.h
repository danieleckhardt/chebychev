#ifndef COEFFICIENTS_H_ 
#define COEFFICIENTS_H_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor_function.h>

#include <cmath>

namespace MavesS // Multi-Waves-Scale
{
namespace Data
{
using namespace dealii;

template <int dim>
class Coefficient_a : public TensorFunction<2,dim> 
{
    public:
        Coefficient_a(double epsilon,  const int example)
        : TensorFunction<2,dim>()
        , epsilon(epsilon) 
        , example(example)
        {}
        virtual Tensor<2,dim> value(const Point<dim> &x ) const override;
    private:
        const double epsilon;
        const int example;
        const double pi = numbers::PI;
};

template <int dim>
class Coefficient_a_matrix_homogenized : public TensorFunction<2,dim> 
{
    public:
        virtual Tensor<2,dim> value(const Point<dim> &x) const override;
    private:
        const double pi = numbers::PI;

};

template <int dim>
class Coefficient_beta  
{
    public:
        double value();
        Tensor<1,dim> value_vec(const Point<dim> &x);
};
} // namespace Data 
} // namespace MavesS
#endif /* COEFFICIENTS_H_ */