#include "Boundary.h"
#include <cmath>

namespace MavesS // Multi-Waves-Scale
{
namespace Data{
using namespace dealii;

    template <int dim, typename number> 
    number BoundaryValuesU<dim,number>::value(const Point<dim> & /*x*/,
                                        const unsigned int component) const
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));
        return number(0);
    }

    template <int dim, typename number> 
    number BoundaryValuesV<dim,number>::value(const Point<dim> & /*x*/,
                                        const unsigned int component) const
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));
        return number(0);
    }


template class BoundaryValuesU<1>;
template class BoundaryValuesV<1>;
template class BoundaryValuesU<2>;
template class BoundaryValuesV<2>;
} // namespace Data
} // namespace MavesS