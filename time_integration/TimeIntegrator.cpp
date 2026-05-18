#include "TimeIntegrator.h"

using namespace dealii;

namespace MavesS //Multi-Waves-Scale
{
using namespace Data;
using namespace Solving;
namespace TimeIntegration
{

template <int dim>
TimeIntegrator<dim>::TimeIntegrator(FE_Q<dim>&  fe_macro, DoFHandler<dim>&  dof_handler_macro,AffineConstraints<double>& constraints_macro, const double & degree_nl_v)
:
fe_macro(fe_macro)
,dof_handler_macro(dof_handler_macro)
,constraints_macro(constraints_macro)
,degree_nl_v(degree_nl_v)
{}

template <int dim>
void TimeIntegrator<dim>::apply_Uboundary_condition(SparseMatrix<double> &system_matrix_macro, Vector<double> &system_rhs_macro, Vector<double>& solution_macro_u)  
{
    constraints_macro.condense(system_matrix_macro, system_rhs_macro);
    BoundaryValuesU<dim> boundary_values_u_function;
    std::map<types::global_dof_index, double> boundary_values0;
    std::map<types::global_dof_index, double> boundary_values1;

    VectorTools::interpolate_boundary_values(dof_handler_macro,
                                            0,
                                            boundary_values_u_function,
                                            boundary_values0);

    MatrixTools::apply_boundary_values(boundary_values0,
                                        system_matrix_macro,
                                        solution_macro_u,
                                        system_rhs_macro);

    // for 1D Problems the right-handside boundary has the index 1.
    if(dim == 1)
    {
        VectorTools::interpolate_boundary_values(dof_handler_macro,
                                                        1,
                                                        boundary_values_u_function,
                                                        boundary_values1);

        MatrixTools::apply_boundary_values(boundary_values1,
                                                    system_matrix_macro,
                                                    solution_macro_u,
                                                    system_rhs_macro);
    }
}
template <int dim>
void TimeIntegrator<dim>::apply_Vboundary_condition(SparseMatrix<double> &system_matrix_macro, Vector<double> &system_rhs_macro, Vector<double>& solution_macro_v)  
{
    constraints_macro.condense(system_matrix_macro, system_rhs_macro);
    BoundaryValuesV<dim> boundary_values_v_function;
    std::map<types::global_dof_index, double> boundary_values0;
    std::map<types::global_dof_index, double> boundary_values1;

    VectorTools::interpolate_boundary_values(dof_handler_macro,
                                            0,
                                            boundary_values_v_function,
                                            boundary_values0);

    MatrixTools::apply_boundary_values(boundary_values0,
                                        system_matrix_macro,
                                        solution_macro_v,
                                        system_rhs_macro);

    // for 1D Problems the right-handside boundary has the index 1.
    if(dim == 1)
    {
        VectorTools::interpolate_boundary_values(dof_handler_macro,
                                                        1,
                                                        boundary_values_v_function,
                                                        boundary_values1);

        MatrixTools::apply_boundary_values(boundary_values1,
                                                    system_matrix_macro,
                                                    solution_macro_v,
                                                    system_rhs_macro);
    }
}

template <int dim>
void TimeIntegrator<dim>::compute_nl_term(const Vector<double> &data, Vector<double> &nl_term)  
{
    nl_term = 0;
    const QGauss<dim> quadrature_formula(fe_macro.degree + 1);
    FEValues<dim>     fe_values(fe_macro,
                            quadrature_formula,
                            update_values | update_JxW_values |
                            update_quadrature_points);

    const unsigned int dofs_per_cell = fe_macro.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    Vector<double>                       local_nl_term(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double>                  data_values(n_q_points);

    for (const auto &cell : dof_handler_macro.active_cell_iterators())
    {
        local_nl_term = 0;
        fe_values.reinit(cell);
        fe_values.get_function_values(data, data_values);
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
        double data_abs = std::abs( data_values[q_point] ) + 0.0001;
        int data_sgn = (data_values[q_point] > 0) - (data_values[q_point] < 0);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            local_nl_term(i) +=
                (data_sgn*(std::pow(data_abs , degree_nl_v) - std::pow(0.0001,degree_nl_v)) *
                fe_values.shape_value(i, q_point) * fe_values.JxW(q_point)); // sgn(u)((|u| + alpha )^degree_nl_v - alpha^degree_nl_v)
        }
        cell->get_dof_indices(local_dof_indices);

        constraints_macro.distribute_local_to_global(local_nl_term,
                                                        local_dof_indices,
                                                            nl_term);
    }
}


template class TimeIntegrator<1>;
template class TimeIntegrator<2>;
}
}
