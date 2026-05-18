#ifndef FUNCTIONS_FOR_MATRIX_H
#define FUNCTIONS_FOR_MATRIX_H

#include <complex>
#include <cmath>
#include <memory>
#include <functional>
#include <vector>

namespace Types{
    template<int dim>
    class Functions_for_Matrix{
        public:
            Functions_for_Matrix();
            // Specify return type for phi1
            std::complex<double> expo(std::complex<double> x);
            std::complex<double> phi1(std::complex<double> x);

            std::vector<double> Nkoeff;
            std::vector<double> Dkoeff;
    };
}


#endif //FUNCTIONS_FOR_MATRIX_H
