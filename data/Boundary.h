#ifndef BOUNDARY_H_ 
#define BOUNDARY_H_

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

#include <cmath>

namespace MavesS // Multi-Waves-Scale
{
namespace Data
{
using namespace dealii;

    template <int dim, typename number = double> //Macro homogeneous Dirichlet boundary conditions
    class BoundaryValuesU : public Function<dim,number>
    {
    public:
        virtual number value(const Point<dim> & p,
                            const unsigned int component = 0) const override;
    };

    template <int dim, typename number = double> 
    class BoundaryValuesV : public Function<dim,number>
    {
    public:
        virtual number value(const Point<dim> & p,
                            const unsigned int component = 0) const override;
    };

}// namespace Data
}// namespace MavesS
#endif /* BOUNDARY_H_ */