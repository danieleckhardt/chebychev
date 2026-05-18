#include  "Functions_for_matrix.h"

namespace Types{
    template<int dim>
    Functions_for_Matrix<dim>::Functions_for_Matrix()
    {
        Nkoeff.resize(7);
        Nkoeff[0] = 1.;
        Nkoeff[1] = 1./26;
        Nkoeff[2] = 5./156;
        Nkoeff[3] = 1./858;
        Nkoeff[4] = 1./5720;
        Nkoeff[5] = 1./205920;
        Nkoeff[6] = 1./8648640;
        Dkoeff.resize(7);
        Dkoeff[0] = 1.;
        Dkoeff[1] = -6./13;
        Dkoeff[2] = 5./52;
        Dkoeff[3] = -5./429;
        Dkoeff[4] = 1./1144;
        Dkoeff[5] = -1./25740;
        Dkoeff[6] = 1./1235520;
    }

    template<int dim>
    std::complex<double> Functions_for_Matrix<dim>::expo(std::complex<double> x)
    {
        return std::exp(x);
    }

    template<int dim>
    std::complex<double> Functions_for_Matrix<dim>::phi1(std::complex<double> x)
    {
        if(std::abs(x)>1e-4)
            return  (exp(x)-1.)/x;
        else if(std::abs(x) > 1e-16){
                std::complex<double> X   = x;
                std::complex<double> Phi = Nkoeff[0] + x * Nkoeff[1];
                std::complex<double> D   = Dkoeff[0] + x * Dkoeff[1];
                for(int k = 2; k<=6; k++)
                {
                        X   = x * X;
                        Phi = Phi + Nkoeff[k] * X;
                        D   = D   + Dkoeff[k] * X;
                }
                return Phi / D;
        }
        else
            return 1.;
    }
    
    template class Functions_for_Matrix<1>;
    template class Functions_for_Matrix<2>;
}

