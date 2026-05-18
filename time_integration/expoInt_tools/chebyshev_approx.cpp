#include "chebyshev_approx.h"

namespace MavesS{
namespace TimeIntegration
{   

template<int dim>
Chebyshev<dim>::Chebyshev(const int degree, Types::Ellipse<dim>& ellipse, double& time_step_size,const SparseMatrix<double>& A, const SparseMatrix<double>& M)
    :
    ellipse(ellipse)
    ,degree(degree)
    ,time_step_size(time_step_size)
    ,A(A)
    ,M(M)
{
    coefficients.resize(degree+1);
}

template <int dim>
double Chebyshev<dim>::scalar_product(
    const Vector<std::complex<double>> &a,
    const Vector<std::complex<double>> &b,
    const SparseMatrix<double> &matrix,
    bool use_matrix)
{
    Vector<std::complex<double>> tmp_vec(a.size());

    if (use_matrix)
    {
        // tmp_vec = A * a, with A real
        Vector<double> tmp_real(a.size());
        Vector<double> tmp_imag(a.size());

        Vector<double> a_real(a.size());
        Vector<double> a_imag(a.size());

        for (unsigned int i = 0; i < a.size(); ++i)
        {
            a_real[i] = a[i].real();
            a_imag[i] = a[i].imag();
        }
        if(a_real.l1_norm()  > 1e-12){
            matrix.vmult(tmp_real, a_real);
            mv_count += 1;
        }
        if(a_imag.l1_norm()  > 1e-12){
            matrix.vmult(tmp_imag, a_imag);
            mv_count += 1;
        }

        for (unsigned int i = 0; i < a.size(); ++i)
            tmp_vec[i] = {tmp_real[i], tmp_imag[i]};
    }
    else
        tmp_vec = a;

    std::complex<double> result = 0.0;

    for (unsigned int i = 0; i < b.size(); ++i)
        result += std::conj(b[i]) * tmp_vec[i];

    return std::abs(result);
}


template<int dim>
void Chebyshev<dim>::get_degree(int& degree_) const
{
    degree_ = this->degree;
}

template<int dim>   
void Chebyshev<dim>::get_coefficients(std::vector<double>& coeffs) const
{
    coeffs = coefficients;
}

template<int dim>
void Chebyshev<dim>::set_degree(int degree_) 
{
    this->degree = degree_;
    coefficients.resize(degree+1);
}

template<int dim>
void Chebyshev<dim>::set_chebyshev_tn_basis(int degree_) 
{
    this->degree = degree_;
    coefficients.assign(degree_ + 1,0.);
    coefficients[degree_] = 0.5; // since in the last step of matix_run the coefficients are multiplied by 2
}

template<int dim>
void Chebyshev<dim>::coefficient_fft(std::function<std::complex<double>(std::complex<double>)> func,std::function<std::complex<double>(std::complex<double>)> transform_func){
    //Generate nodes 
    std::vector<std::complex<double>> nodes(2*degree);
    for (int i = 0; i < 2*degree; ++i) {
        nodes[i] = std::complex<double>((2.0 * M_PI * i) / (2*degree));  // Equally spaced in [0, 2π)
    }

    // Evaluate the function at transformed nodes
    std::vector<std::complex<double>> int_func(2*degree);
    for (int i = 0; i < 2*degree; ++i) {
        int_func[i] = func(transform_func(nodes[i]));
    }

    //Compute FFT using FFTW
    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 2*degree);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 2*degree);
    fftw_plan plan = fftw_plan_dft_1d(2*degree, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    
    // Fill input (real part only, imaginary part = 0)
    for (int i = 0; i < 2*degree; ++i) {
        in[i][0] = int_func[i].real();
        in[i][1] = int_func[i].imag();
    }

    // Execute FFT
    fftw_execute(plan);

    // Extract coefficients and normalize
    for (int k = 0; k < degree ; ++k) {
        coefficients[k] = out[k][0] / (2*degree);
    }

    // Cleanup FFTW resources
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}

template<int dim>
std::complex<double> Chebyshev<dim>::scalar_run(std::function<std::complex<double>(std::complex<double>)> func, std::complex<double> x ){

    double radiusx; double radiusy; double center;
    ellipse.get_radx(radiusx); 
    ellipse.get_rady(radiusy);
    ellipse.get_center(center);

    std::complex<double> factor = (radiusx - radiusy) / (radiusx + radiusy);
    std::complex<double> inv_scale = 1. / std::sqrt(std::complex<double>(radiusx * radiusx - radiusy * radiusy));


    auto transform_func = [radiusx,radiusy, center](std::complex<double> z) {
        return center + std::cos(z)*radiusx + 1i*std::sin(z)*radiusy;
        };  

    std::complex<double> x_trans = (x - center) * inv_scale; // (x - center)/(a**2 - b**2)
    coefficient_fft(func,transform_func);  

    
    std::complex<double> cheb_np1;
    std::complex<double> cheb_n_1 = 1.;
    std::complex<double> result = coefficients[0]; // c0*x
    //
    std::complex<double> cheb_n = std::sqrt(factor)*x_trans;
    result += 2*coefficients[1]* cheb_n;
    //
    for(unsigned int k = 2; k < coefficients.size(); ++k ){
        cheb_np1 = cheb_n;
        cheb_n *= std::sqrt(factor)*x_trans; 
        cheb_n *= 2;
        cheb_n -= factor * cheb_n_1; 
        //
        cheb_n_1 = cheb_np1;
        result +=  2*coefficients[k]*cheb_n ;
    }
    return result;
}

template<int dim>
std::complex<double> Chebyshev<dim>::scalar_run( std::complex<double> x ){

    double radiusx; double radiusy; double center;
    ellipse.get_radx(radiusx); 
    ellipse.get_rady(radiusy);
    ellipse.get_center(center);

    // std::complex<double> factor = (radiusx - radiusy) / (radiusx + radiusy);
    std::complex<double> inv_scale = 1. / std::sqrt(std::complex<double>(radiusx * radiusx - radiusy * radiusy));

    std::complex<double> x_trans = (x - center) * inv_scale; // (x - center)/(a**2 - b**2)
    
    std::complex<double> cheb_np1;
    std::complex<double> cheb_n_1 = 1.;
    std::complex<double> result = coefficients[0]; // c0*x
    //
    std::complex<double> cheb_n = x_trans;
    result += 2*coefficients[1]* cheb_n;
    //
    for(unsigned int k = 2; k < coefficients.size(); ++k ){
        cheb_np1 = cheb_n;
        cheb_n *= x_trans; 
        cheb_n *= 2;
        cheb_n -=  cheb_n_1; 
        //
        cheb_n_1 = cheb_np1;
        result +=  2*coefficients[k]*cheb_n ;
    }
    return result;
}

template<int dim>
void Chebyshev<dim>::estimate_degree_ellipse_by_estimate(int phi_k, double tol)
{
    double radiusx, radiusy, center;
    ellipse.get_radx(radiusx);
    ellipse.get_rady(radiusy);
    ellipse.get_center(center);

    std::complex<double> factor = std::sqrt(std::complex<double>((radiusx + radiusy) / (radiusx - radiusy)));
    std::complex<double> omega  = std::sqrt(std::complex<double>(radiusx*radiusx - radiusy*radiusy));

    std::function<double(int)> reference_func;

    // if (omega.real() != 0.0)
    // {
    //     double c = 1.2;
    //     double constant_0 = std::sqrt(20.0/M_PI) *
    //                         std::pow(4.0, phi_k) *
    //                         std::tgamma(phi_k + 1) /
    //                         std::tgamma(2*phi_k + 1);

    //     auto constant_first =
    //         [=](int m) {
    //             double denom = m - c*radiusy;
    //             if (std::abs(denom) < 1e-14) return HUGE_VAL;
    //             return constant_0 * c * std::sqrt(radiusx) / denom;
    //         };

    //     auto constant_second =
    //         [=]() {
    //             return constant_0 *
    //                    std::exp(omega.real()/4 - radiusx) /
    //                    std::sqrt(radiusx);
    //         };

    //     auto expo_first =
    //         [=](int m) {
    //             return ((2*c*radiusy - m)*m - (c*c*radiusy*radiusy)) /
    //                    (2*c*radiusx);
    //         };

    //     auto expo_second =
    //         [=](int m) {
    //             return std::exp(1.0) * (radiusx + radiusy) / (2*m);
    //         };

    //     auto small_reference =
    //         [=](int m) {
    //             // std::cout << " 1 ";
    //             return constant_first(m) * std::exp(expo_first(m));
    //         };

    //     auto large_reference =
    //         [=](int m) {
    //             // std::cout << " 2 " <<  std::exp(1.0) * (radiusx + radiusy) / (2*m) << " ";
    //             double log_val = m * std::log(expo_second(m));
    //             // std::cout << "log " << log_val << " ";
    //             if (log_val < -700)   // double underflow threshold
    //                 return 0.0;
    //             return constant_second() * std::exp(log_val);
    //         };

    //     reference_func = [=](int m) {
    //         if (m > c*radiusx + 1 && m < c*(radiusx + radiusy))
    //             return small_reference(m);
    //         else if (m >= c*(radiusx + radiusy))
    //             return large_reference(m);
    //         else
    //             return 1.0;
    //     };
    // }
    // else
    // {
    //     double constant =
    //         4/std::tgamma(phi_k + 1) * (std::abs(factor) + 1.0) * std::exp(-radiusx);


    //     auto expo = [=](int m) {
    //         if (m <= 0) return HUGE_VAL;
    //         return std::exp(1 - std::abs(omega*omega)/(4.0*m*m)) *
    //                (radiusx + radiusy) / (2*m);
    //     };
    //     double c = std::abs((factor + 1.)/(2.*factor + 1.));
    //     reference_func = [=](int m) {
    //         if (m >= c*(radiusx + radiusy)){
    //             return constant * std::pow(expo(m), m);
    //         }
    //         else{
    //             return 1.0;
    //         }
    //     };
    // }

    double constant =
            4*std::sqrt(M_PI)/std::sqrt(2)  * std::exp(radiusx + center);
    auto expo = [=](int m) {
            if (m <= 0) return HUGE_VAL;
            return  std::exp(1.0)*(radiusx + radiusy) / (2*m);
        };
    reference_func = [=](int m) {
            if (m >= (radiusx + radiusy)){
                return constant *1/( std::sqrt(m) * std::pow(m, phi_k))* std::pow(expo(m), m);
            }
            else{
                return 1.0;
            }  
    };

    // Build seek function
    auto seek_degree = [&](int m) {
        return reference_func(m) - tol;
    };

    // Binary search
    // std::cout << "Binary search for degree..." << std::endl;
    int left_m =  1.2*radiusy + 1;
    int right_m = 1.2*(radiusx + radiusy) + 1;
    if (seek_degree(left_m) < 0)
    {
        // Decrease until root bracketed
        while (seek_degree(left_m) < 0 && left_m > 1)
            left_m /= 2;

        right_m = left_m * 2 + 1;
    }
    else if (seek_degree(left_m) > 0 && seek_degree(right_m) > 0)
    {
        // Expand until root bracketed
        while (seek_degree(right_m) > 0){
            // std::cout << " Expanding since " << "(reference_func - tol)("  << right_m << ") = " <<  seek_degree(right_m) << " > 0" << std::endl;
            right_m *= 2;
        }

        left_m = right_m / 2;
    }

    // Now root is bracketed
    while (right_m - left_m > 1)
    {
        int mid = (left_m + right_m) / 2;
        double fmid = seek_degree(mid);

        if (std::abs(fmid) < 1e-14)
        {
            degree = mid;
            std::cout << "Lucky hit! Degree = " << degree << ", Error = " << seek_degree(degree) + tol << " " << std::endl;
            return;
        }

        if (seek_degree(left_m) * fmid < 0)
            right_m = mid;
            
        else{
            // std::cout << " Degree = " << mid << ", Error mid = " << seek_degree(mid) + tol << " " << std::endl;
            left_m = mid;
        }
    }
    degree = std::max(right_m-1,1);
    std::cout << " Degree = " << degree << ", Error = " << seek_degree(degree) + tol << " " << std::endl;
    coefficients.resize(degree);

}


template<int dim>
void Chebyshev<dim>::estimate_degree_ellipse_by_error(std::function<std::complex<double>(std::complex<double>)> func, double tol_, int max_iter ) 
{
    /*
    Estimate the degree of the Chebyshev approximation on an ellipse.
    */
    int i = 0; int max_degree = 1000; // Maximum degree to prevent infinite loops
    double err;
    while (i < max_iter) {
        // Define lambda to approximate f using Chebyshev series
        auto f_approx = [this, func](std::complex<double> z) {
            return scalar_run(func, z);
        };

        err = ellipse.calc_error(func, f_approx);

        if ((!std::isnan(err) && err < tol_) || max_degree < degree) {
            // std::cout << "Convergence achieved with error: " << err ;
            break;
        }
        // Update degree
        degree = static_cast<int>(degree * (1 + std::log(1 + std::min(1.0, err)))) + std::max(1, int(std::abs(std::log(tol_))*log(1 + err) ));
        std::cout << "Iteration " << i << ": Degree = " << degree << ", Error = " << err << " " << std::endl;
        coefficients.resize(degree);
        i++;
    }
}

template <int dim>
double Chebyshev<dim>::calculate_residual(int k_phi)
{
    // Compute column-wise L2 norms of temp = Matrix_x_vector - vector_primes
    res = Matrix_x_vector;
    res.add(-1.0, vector_primes); 

    Vector<std::complex<double>> res_u(res.m());
    Vector<std::complex<double>> res_v(res.m());
    Vector<std::complex<double>> Ma_u(res.m());
    Vector<std::complex<double>> Ma_v(res.m());
    double l2_norm_tmp = 0.;

    for(unsigned long i = 0; i < res.m(); ++i){
        res_u(i) = res(i,0);
        res_v(i) = res(i,1);
        Ma_u(i)     = Matrix_x_vector(i,0);
        Ma_v(i)     = Matrix_x_vector(i,1);
        // l2_norm_tmp += std::norm(Matrix_x_vector(i,0)) + std::norm(Matrix_x_vector(i,1));
    }
    l2_norm_tmp = std::sqrt(scalar_product(Ma_u,Ma_u,A) + scalar_product(Ma_v,Ma_v,M));
    l2_norm_tmp = std::min(l2_norm_tmp,1.); // avoid to early stops due to very small norms
    double residual = std::sqrt(scalar_product(res_u,res_u,A) + scalar_product(res_v,res_v,M));
    return std::pow(time_step_size,k_phi-1) * residual/l2_norm_tmp;
}


template<int dim>
void Chebyshev<dim>::apply_scaled_chebyshev(
    BMatrixVariant Matrix_mult,
    const std::complex<double> &factor,
    double &center,
    const std::complex<double> &inv_scale,
    std::complex<double> &scaling_ref,
    std::complex<double> &scaling_new,
    std::complex<double> &scaling_old,
    FullMatrix<std::complex<double>> &blockvector)
{
    cheb_n_1 = blockvector;
    blockvector *= coefficients[0]; // c0 * x

    cheb_n = cheb_n_1;
    mv_count = std::get<std::function<int(FullMatrix<std::complex<double>>&)>>(Matrix_mult)(cheb_n);
    cheb_n.add(-center, cheb_n_1);
    cheb_n *= std::sqrt(factor) * inv_scale;
    cheb_n *= scaling_new; // with scaling

    blockvector.add(2.0 * coefficients[1], cheb_n);

    for (unsigned int k = 2; k < coefficients.size(); ++k) {
        scaling_new = 1. / (2. / scaling_ref - scaling_old);
        cheb_np1 = cheb_n;

        mv_count += std::get<std::function<int(FullMatrix<std::complex<double>>&)>>(Matrix_mult)(cheb_np1);
        cheb_np1.add(-center, cheb_n);
        cheb_np1 *= 2.0 * scaling_new * std::sqrt(factor) * inv_scale;
        cheb_np1.add(-factor * scaling_new * scaling_old, cheb_n_1);
        scaling_old = scaling_new;

        cheb_n_1 = cheb_n;
        cheb_n = cheb_np1;
        blockvector.add(2.0 * coefficients[k], cheb_n);
    }
}

template<int dim>
void Chebyshev<dim>::apply_scaled_chebyshev_alternativ(
    BMatrixVariant Matrix_mult,
    double &center,
    double &radiusx,
    double &radiusy,
    FullMatrix<std::complex<double>> &blockvector)
{
    double factor = (radiusx - radiusy) / (radiusx + radiusy);
    double scaling = 1/(radiusx + radiusy);
    cheb_n_1 = blockvector;
    cheb_n_1 *= 2;
    blockvector *= 2*coefficients[0]; // c0 * x

    cheb_n = cheb_n_1;
    mv_count = std::get<std::function<int(FullMatrix<std::complex<double>>&)>>(Matrix_mult)(cheb_n);
    cheb_n.add(-center, cheb_n_1);
    cheb_n *= scaling;

    blockvector.add(2.0 * coefficients[1], cheb_n);

    for (unsigned int k = 2; k < coefficients.size(); ++k) {
        cheb_np1 = cheb_n;

        mv_count += std::get<std::function<int(FullMatrix<std::complex<double>>&)>>(Matrix_mult)(cheb_np1);
        cheb_np1.add(-center, cheb_n);
        cheb_np1 *= 2.0 * scaling ;
        cheb_np1.add(-factor , cheb_n_1);

        cheb_n_1 = cheb_n;
        cheb_n = cheb_np1;
        blockvector.add(2.0 * coefficients[k], cheb_n);
    }
}

template<int dim>
void Chebyshev<dim>::apply_residual_chebyshev(
    BMatrixVariant Matrix_mult,
    const std::complex<double> &factor,
    double &center,
    double &radiusx,
    double &radiusy,
    const std::complex<double> &inv_scale,
    FullMatrix<std::complex<double>> &blockvector,
    double tol,
    int k_phi)
{
    cheb_n_3 = blockvector;
    cheb_n_3 *= -1./factor;
    cheb_n_2 = blockvector;
    cheb_n_2 *= 0.;
    cheb_n_1 = blockvector;
    Matrix_x_vector = blockvector;
    vector_primes   = blockvector;
    vector_primes   *= 0.; 
    //
    blockvector *= coefficients[0];
    int fac = k_phi;
    if (fac > 1){
        for (int i = 1; i <= k_phi-1; ++i) {
            fac *= i;
        }
    }
    Matrix_x_vector *= (fac - k_phi*coefficients[0]);
    // Matrix_x_vector *= 0.; // test

    cheb_n = cheb_n_1;
    mv_count = std::get<std::function<int(FullMatrix<std::complex<double>>&)>>(Matrix_mult)(cheb_n);
    Matrix_x_vector.add(coefficients[0],cheb_n);

    cheb_n.add(-center, cheb_n_1);
    cheb_n *= 2.0 * std::sqrt(factor) * inv_scale; // second kind scaling

    for (unsigned int k = 1; k < coefficients.size()-1; ++k) {
        cheb_np1 = cheb_n;
        mv_count += std::get<std::function<int(FullMatrix<std::complex<double>>&)>>(Matrix_mult)(cheb_np1);
        cheb_np1.add(-center, cheb_n);
        cheb_np1 *= 2.0 * std::sqrt(factor) * inv_scale;
        cheb_np1.add(-factor, cheb_n_1);
        // compute new approximations
        blockvector.add(coefficients[k], cheb_n);
        blockvector.add(-coefficients[k]*factor, cheb_n_2);

        Matrix_x_vector.add(0.5*coefficients[k]*(radiusx + radiusy),cheb_np1);
        Matrix_x_vector.add(coefficients[k]*(center - float(k_phi)),cheb_n);
        Matrix_x_vector.add(-coefficients[k]*(center - float(k_phi) )*factor,cheb_n_2);
        Matrix_x_vector.add(-0.5*coefficients[k]*factor*(radiusx - radiusy),cheb_n_3);

        vector_primes.add(std::complex<double>(k)*coefficients[k], cheb_n);
        vector_primes.add(std::complex<double>(k)*2.*center/(radiusx + radiusy)*coefficients[k], cheb_n_1);
        vector_primes.add(std::complex<double>(k)*factor*coefficients[k], cheb_n_2);
        //
        double res =  calculate_residual(k_phi);
        if(res < tol){
            std::cout << "  Chebyshev converged:" 
                        << " degree: "           << k
                        // << ", Max degree: "      << coefficients.size()
                        << ", Residual: "        << res 
                        << std::endl;
            set_degree(k);
            break;
        }
        // Update 
        cheb_n_3 = cheb_n_2;
        cheb_n_2 = cheb_n_1;
        cheb_n_1 = cheb_n;
        cheb_n = cheb_np1;
    }
}

template<int dim>
void Chebyshev<dim>::matrix_run(
    BMatrixVariant Matrix_mult,
    std::function<std::complex<double>(std::complex<double>)> func,
    FullMatrix<std::complex<double>> & blockvector,
    const std::optional<std::complex<double>> &ref_point_opt, /* optional parameter */
    double tol,
    int k_phi
)
{
    // Common ellipse setup
    double radiusx, radiusy, center;
    ellipse.get_radx(radiusx);
    ellipse.get_rady(radiusy);
    ellipse.get_center(center);

    if (std::abs(radiusx + radiusy) < 1e-10)
        throw std::invalid_argument("a + b must not be zero to avoid division by zero.");

    std::complex<double> inv_scale = 1. /
        std::sqrt(std::complex<double>(radiusx * radiusx - radiusy * radiusy));

    auto transform_func = [radiusx, radiusy, center](std::complex<double> x) {
        return center + std::cos(x) * radiusx + 1i * std::sin(x) * radiusy;
    };

    // Compute coefficients
    std::complex<double> factor;
    bool use_scaling = ref_point_opt.has_value();

    if (!use_scaling) {
        coefficient_fft(func, transform_func);
        factor = (radiusx - radiusy) / (radiusx + radiusy);
    } else {
        factor = std::complex<double>(1.0, 0.0);
    }

    // Optional scaling variables
    std::complex<double> ecc, scaling_ref, scaling_new, scaling_old;
    ellipse.get_ecc(ecc);
    if (use_scaling) {
        scaling_ref = ecc / (*ref_point_opt - center);
        scaling_new = scaling_old = scaling_ref;
    }

    // --- Dispatch to appropriate routine ---
    if (use_scaling)
        apply_scaled_chebyshev(Matrix_mult, factor, center, inv_scale, scaling_ref, scaling_new, scaling_old, blockvector);
        // apply_scaled_chebyshev_alternativ(Matrix_mult,center,radiusx,radiusy, blockvector);
    else
        apply_residual_chebyshev(Matrix_mult, factor, center, radiusx, radiusy, inv_scale, blockvector, tol, k_phi);
}

template class Chebyshev<1>;
template class Chebyshev<2>;

}
}