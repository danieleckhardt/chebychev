#ifndef INITIALCONDITIONS_H_ 
#define INITIALCONDITIONS_H_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"


#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

#include <math.h>

namespace MavesS // Multi-Waves-Scale
{
namespace Data
{
using namespace dealii;

    template <int dim, typename number = double>
    class InitialCondition : public Function<dim,number>
    {
    public:
    enum Components
    {
      U, 
      V
    };

    InitialCondition(const Components component_init, const int example)
        : Function<dim,number>()
        ,component_init(component_init)
        ,example(example)
    {}


    virtual number value(const Point<dim> & x,
                            const unsigned int component = 0) const override;
    private:
    const Components component_init;
    const int example;
    int gaus = 300;


    };

}// namespace Data
}// namespace MavesS
#endif /* INITIALCONDITIONS_ */