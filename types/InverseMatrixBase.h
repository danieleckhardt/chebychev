/*
 * InverseMatrixBase.h
 *
 *  Created on: Feb 14, 2018
 *      Author: leibold
 */
 
#ifndef DATA_TYPES_INVERSEMATRIXBASE_H_
#define DATA_TYPES_INVERSEMATRIXBASE_H_
 
#include <deal.II/lac/vector.h>
 
using namespace dealii;
 
namespace Types{
 
template<int dim, typename number = double>
class InverseMatrixBase {
public:
 
        /**
         *
         * Virtual destructor
         */
        virtual ~InverseMatrixBase() {};
 
        /**
         *
     * This routine computes one complete time step with the time integration scheme.
         */
        virtual unsigned int vmult(Vector<number> &dst,  Vector<number> &src, double tol_) = 0;
        virtual unsigned int vmult_mass_lumping(Vector<number> &dst,   const Vector<number> &src) =0;
        virtual unsigned int mmult(FullMatrix<number>& C, const FullMatrix<number>& B, double tol) = 0; 
};
 
} /* namespace Types */
 
#endif /* DATA_TYPES_INVERSEMATRIXBASE_H_ */