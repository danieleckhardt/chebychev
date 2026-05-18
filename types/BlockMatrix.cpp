#include "BlockMatrix.h"

namespace Types{


template <int dim>
BlockMatrix<dim>::BlockMatrix(SparseMatrix<double>& A,
                              SparseMatrix<double>& B,
                              SparseMatrix<double>& M,
                              DoFHandler<dim>& dof_handler,
                              double factor,
                              bool apply_bd_conditions,
                              bool sym_damp,
                              bool use_pre_for_mass
                            )
    : A(A)
    , B(B)
    , M(M)
    , rat_krylov(M.get_sparsity_pattern())
    , apply_bd_conditions(apply_bd_conditions)
    , sym_damp(sym_damp)
    , factor(factor)
    , dof_handler(dof_handler)
    , M_inv(M, dof_handler, apply_bd_conditions,true,use_pre_for_mass)
    , M_inv_dealii(M, dof_handler, apply_bd_conditions)
    , size(dof_handler.n_dofs())
{
    // rat_krylov.reinit(M.get_sparsity_pattern());
    update_rat_krylov_matrix();
    if(use_pre_for_mass) {
        cost_pre = 2;
    }
}

// === set_factor Method ===
template <int dim>
void BlockMatrix<dim>::set_factor(double new_factor) {
    factor = new_factor;
    update_rat_krylov_matrix();
    size = dof_handler.n_dofs(); // Optional: may not be needed unless size can change
}

// === Private helper method ===
template <int dim>
void BlockMatrix<dim>::update_rat_krylov_matrix() {
    rat_krylov.copy_from(M);
    rat_krylov.add(factor * rat_shift, B);
    rat_krylov.add(factor * factor * rat_shift * rat_shift, A);

    rat_krylov_inv = std::make_unique<InverseMatrixPETSc<dim, double>>(
        rat_krylov,  dof_handler, apply_bd_conditions, sym_damp);
}

template<int dim> 
int BlockMatrix<dim>::get_size(){
    return size;
}
template<int dim>
void BlockMatrix<dim>::get_matrices(const SparseMatrix<double>*& A_out,
                                    const SparseMatrix<double>*& B_out,
                                    const SparseMatrix<double>*& M_out,
                                    const InverseMatrix<dim, double>*& M_inv_out) {
    A_out     = &this->A;
    B_out     = &this->B;
    M_out     = &this->M;
    M_inv_out = &this->M_inv_dealii;
}

template<int dim>
int BlockMatrix<dim>::vmult(Vector<double> &a,  Vector<double> &b, double tol){
    old_a = a;
    old_b = b;
    //
    A.vmult(a,old_a);
    B.vmult(b,old_b);
    int number_mv = 2;
    b += a;
    b *= -factor ;
    double norm = b.l2_norm();
    number_mv += cost_pre*M_inv.vmult(b, b,tol*norm);  // 2* when use preconditioner

    // 
    a = old_b; 
    a *= factor; 
    return number_mv;
}

template<int dim>
int BlockMatrix<dim>::vmult_shift(Vector<double> &a,  Vector<double> &b, double tol){
    old_a = a;
    old_b = b;
    //
    int number_mv = vmult(a,b,tol);
    //
    b *= rat_shift*factor;
    b -= old_b ;
    b *= -1;
    a *= rat_shift*factor;
    a -= old_a ;
    a *= -1;
    //
    return number_mv;
}

template<int dim>
int BlockMatrix<dim>::vmult_weak_op(Vector<double> &a,  Vector<double> &b){
    old_a = a;
    old_b = b;
    //
    A.vmult(a,old_a);
    B.vmult(b,old_b);
    b += a;
    b *= -factor ;
    // 
    A.vmult(a,old_b); 
    a *= factor; 
    return 3;
}

template<int dim>
int BlockMatrix<dim>::vmult(Vector<double> &a,  Vector<double> &b, const Vector<double> &old_a,  const Vector<double> &old_b, double tol){
    //
    A.vmult(a,old_a);
    B.vmult(b,old_b);
    int number_mv = 2;
    b += a;
    b *= -factor ;
    double norm = b.l2_norm();
    number_mv += cost_pre*M_inv.vmult(b, b,tol*norm);  // 2* since we use preconditioner

    // 
    a = old_b; 
    a *= factor; 
    return number_mv;
}

template<int dim>
int BlockMatrix<dim>::vmult_weak_op(Vector<double> &a,  Vector<double> &b, const Vector<double> &old_a,  const Vector<double> &old_b){
    //
    A.vmult(a,old_a);
    B.vmult(b,old_b);
    b += a;
    b *= -factor ;
    // 
    A.vmult(a,old_b); 
    a *= factor; 
    return 3;
}

template<int dim>
int BlockMatrix<dim>::vmult_rat(Vector<double> &a,  Vector<double> &b, const Vector<double> &old_a,  const Vector<double> &old_b, double tol){
    //
    A.vmult(a,old_a);
    M.vmult(b,old_b);
    int number_mv = 2;
    a *= factor*rat_shift;
    b -= a;
    double norm = b.l2_norm();
    number_mv += 2*rat_krylov_inv->vmult(b,b,tol*norm); //2* since we use preconditioner
    //
    a = b;
    a *= factor*rat_shift;
    a += old_a;
    return number_mv;
    //
}

template<int dim>
int BlockMatrix<dim>::vmult_rat(Vector<double> &a,  Vector<double> &b,  double tol){
    //
    Vector<double> old_a(a);
    Vector<double> old_b(b);
    return vmult_rat(a,b,old_a,old_b,tol);
    //

}

template<int dim>
void BlockMatrix<dim>::vmultT(Vector<double> &a,  Vector<double> &b, const Vector<double> &old_a,  const Vector<double> &old_b, double tol){
    //
    double number_mv = 2;
    B.vmult(b,old_b);
    double norm = b.l2_norm();
    number_mv += cost_pre*M_inv.vmult(b, b,tol*norm);    
    b *= -1 ;
    b += old_a;
    b *= factor;
    // 
    A.vmult(a,old_b);
    norm = b.l2_norm();
    number_mv += cost_pre*M_inv.vmult(b, b,tol*norm);  
    a *= factor; 
}

template<int dim>
int BlockMatrix<dim>::vmult_complex_split(Vector<std::complex<double>> &a,  Vector<std::complex<double>> &b, double tol, bool rational){
    const unsigned int size = a.size();
    int number_mv = 0;
    a_real.reinit(size), a_imag.reinit(size);
    b_real.reinit(size), b_imag.reinit(size);
    for (unsigned int i = 0; i < size; ++i)
    {
        a_real[i] = a[i].real();
        a_imag[i] = a[i].imag();
        b_real[i] = b[i].real();
        b_imag[i] = b[i].imag();
    }
    
    if(rational){
        if(a_real.l1_norm() + b_real.l1_norm() > 1e-12 )
            number_mv +=  vmult_rat(a_real,b_real, tol);
        if(a_imag.l1_norm() + b_imag.l1_norm() > 1e-12 )
            number_mv +=  vmult_rat(a_imag,b_imag, tol);
    }
    else{
        if(a_real.l1_norm() + b_real.l1_norm() > 1e-12 )
            number_mv +=  vmult(a_real,b_real, tol);
        if(a_imag.l1_norm() + b_imag.l1_norm() > 1e-12 )
            number_mv +=  vmult(a_imag,b_imag, tol);
    }

    for (unsigned int i = 0; i < size; ++i){
        a[i] = std::complex<double>(a_real[i],a_imag[i]);
        b[i] = std::complex<double>(b_real[i],b_imag[i]);
    }
    return number_mv;

}

template<int dim>
int BlockMatrix<dim>::vmult_complex_split(Vector<std::complex<double>> &a,  Vector<std::complex<double>> &b, const Vector<std::complex<double>> &old_a,  const Vector<std::complex<double>> &old_b, double tol){
    const unsigned int size = a.size();
    int number_mv = 0;
    a_real.reinit(size), a_imag.reinit(size);
    b_real.reinit(size), b_imag.reinit(size);
    for (unsigned int i = 0; i < size; ++i)
    {
        a_real[i] = old_a[i].real();
        a_imag[i] = old_a[i].imag();
        b_real[i] = old_b[i].real();
        b_imag[i] = old_b[i].imag();
    }

    number_mv += vmult(a_real,b_real, tol);
    number_mv += vmult(a_imag,b_imag, tol);

    for (unsigned int i = 0; i < size; ++i){
        a[i] = std::complex<double>(a_real[i],a_imag[i]);
        b[i] = std::complex<double>(b_real[i],b_imag[i]);
    }
    return number_mv;

}

template<int dim>
int BlockMatrix<dim>::vmult_complex_split_weak_op(Vector<std::complex<double>> &a,  Vector<std::complex<double>> &b, const Vector<std::complex<double>> &old_a,  const Vector<std::complex<double>> &old_b){
    const unsigned int size = a.size();
    int number_mv = 0;
    a_real.reinit(size), a_imag.reinit(size);
    b_real.reinit(size), b_imag.reinit(size);
    for (unsigned int i = 0; i < size; ++i)
    {
        a_real[i] = old_a[i].real();
        a_imag[i] = old_a[i].imag();
        b_real[i] = old_b[i].real();
        b_imag[i] = old_b[i].imag();
    }

    number_mv += vmult_weak_op(a_real,b_real);
    number_mv += vmult_weak_op(a_imag,b_imag);

    for (unsigned int i = 0; i < size; ++i){
        a[i] = std::complex<double>(a_real[i],a_imag[i]);
        b[i] = std::complex<double>(b_real[i],b_imag[i]);
    }
    return number_mv;

}

template<int dim>
int BlockMatrix<dim>::vmultBlock(FullMatrix<std::complex<double>>& block_vec, double tol ){
    Vector<std::complex<double>> a(block_vec.m());
    Vector<std::complex<double>> b(block_vec.m());
    for(unsigned long i = 0; i < block_vec.m(); ++i){
        a(i) = block_vec(i,0);
        b(i) = block_vec(i,1);
    }

    int number_mv = vmult_complex_split(a,b,tol,false); //faster 

    for(unsigned long i = 0; i < block_vec.m(); ++i){
        block_vec(i,0) = a(i);
        block_vec(i,1) = b(i);
    }
    return number_mv;
}


template class BlockMatrix<1>;
template class BlockMatrix<2>;
}/* namespace Types */