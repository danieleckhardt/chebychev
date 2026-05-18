#include "Ellipse.h"

namespace Types{

template<int dim> 
Ellipse<dim>::Ellipse(double radiusx, std::complex<double> eccentricity, double center )
:
    radiusx(radiusx)
    ,center(center)
    ,eccentricity(eccentricity)
{
    if(radiusx < eccentricity.real()){
        throw std::invalid_argument("radiusx should be bigger the the real part of eccentricity");
    }
    std::complex<double> radiusy_temp = std::sqrt( radiusx*radiusx - eccentricity*eccentricity);
    radiusy = radiusy_temp.real();

    radius_ellipse = radius_p(center + radiusx);
    volume         = 3.1415926535897932*radiusx*radiusy;
    
}

template<int dim> 
void Ellipse<dim>::get_parameters( double& radiusx, double& radiusy, std::complex<double>& eccentricity, double& center ){
    radiusx = this->radiusx;
    radiusy = this->radiusy;
    eccentricity = this->eccentricity;
    center = this->center;
}
template<int dim> 
void Ellipse<dim>::get_radx( double& radiusx){
    radiusx = this->radiusx;
}
template<int dim> 
void Ellipse<dim>::get_rady( double& radiusy){
    radiusy = this->radiusy;
}
template<int dim> 
void Ellipse<dim>::get_ecc(std::complex<double>& eccentricity){
    eccentricity = this->eccentricity;
}
template<int dim> 
void Ellipse<dim>::get_center(double& center){
    center = this->center;
}
template<int dim>
void Ellipse<dim>::get_radius_ellipse(double& radius_ell){
    radius_ell = this->radius_ellipse;
}
template<int dim>
void Ellipse<dim>::get_volume_ellipse(double& radius_vol){
    radius_vol = this->volume;
}
//
template<int dim> 
void Ellipse<dim>::set_parameters( double radiusx, double radiusy, double center ){
    this->radiusx = radiusx ;
    this->radiusy = radiusy ;
    this->center = center;
    eccentricity = sqrt( std::complex<double>( this->radiusx*this->radiusx -  this->radiusy* this->radiusy) );
    radius_ellipse = radius_p(center + radiusx);
    volume   = 3.1415926535897932*radiusx*radiusy;

}
template<int dim> 
void Ellipse<dim>::set_radx( double radiusx){
    this->radiusx = radiusx;
    eccentricity = sqrt( std::complex<double>( this->radiusx*this->radiusx - radiusy*radiusy) );
    radius_ellipse = radius_p(center + radiusx);
    volume   = 3.1415926535897932*radiusx*radiusy;

}

template<int dim> 
bool Ellipse<dim>::set_radx_with_fix_center_and_ecc( double radiusx){
    if(eccentricity.imag() == 0. && radiusx < eccentricity.real()){
        this->radiusx = eccentricity.real();
        radiusy = 0.;
        radius_ellipse = radius_p(center + this->radiusx);
        volume   = 0.;
        return false;
    }
    else if (radiusx < 0.)
    {
        radiusy = eccentricity.imag();
        this->radiusx = 0.;
        radius_ellipse = radius_p(center + 1i*this->radiusy);
        volume   = 0.;
        return false;
    }
    else{
        this->radiusx = radiusx;
        radiusy = std::abs(sqrt( std::complex<double>( this->radiusx*this->radiusx - eccentricity*eccentricity) ));
        radius_ellipse = radius_p(center + radiusx);
        volume   = 3.1415926535897932*radiusx*radiusy;

        return true;
    }
}

template<int dim> 
bool Ellipse<dim>::set_rady_with_fix_center_and_ecc( double radiusy){
    if(eccentricity.real() == 0. && radiusy < eccentricity.imag()){
        this->radiusy = eccentricity.imag();
        radiusx = 0.;
        radius_ellipse = radius_p(center + 1i*this->radiusy);
        volume   = 0.;
        return false;
    }
    else if (radiusy <= 0.)
    {
        radiusx = eccentricity.real();
        this->radiusy = 0.;
        radius_ellipse = radius_p(center + this->radiusx);
        volume   = 0.;
        return false;
    }
    else{
        this->radiusy = radiusy;
        radiusx = std::abs(sqrt( std::complex<double>(  eccentricity*eccentricity + this->radiusy*this->radiusy ) ));
        radius_ellipse = radius_p(center + radiusx);
        volume   = 3.1415926535897932*radiusx*radiusy;
        return true;
    }
}

template<int dim> 
void Ellipse<dim>::set_rady( double radiusy){
    this->radiusy = radiusy;
    eccentricity = sqrt( std::complex<double>( radiusx*radiusx - this->radiusy*this->radiusy) );
    radius_ellipse = radius_p(center + radiusx);
    volume   = 3.1415926535897932*radiusx*radiusy;

}
template<int dim> 
void Ellipse<dim>::set_ecc(std::complex<double> eccentricity){
    this->eccentricity = eccentricity;
    if (eccentricity.real() != 0){
        radiusx = std::abs(std::sqrt( this->eccentricity*this->eccentricity + std::complex<double>( radiusy*radiusy)));
    }
    if (eccentricity.imag() != 0){
        radiusy = std::abs(std::sqrt( std::complex<double>( radiusx*radiusx) - this->eccentricity*this->eccentricity));
    }
    radius_ellipse = radius_p(center + radiusx);
    volume   = 3.1415926535897932*radiusx*radiusy;


}
template<int dim> 
void Ellipse<dim>::set_center(double center){
    this->center = center;
    radius_ellipse = radius_p(center + radiusx);

}

template<int dim> 
double Ellipse<dim>::findRealRootBoost(const std::vector<double>& coeffs, double max , double min , double step ) {
    if (step <= 0) {
        throw std::invalid_argument("Step size must be positive.");
    }

    if (max < min) std::swap(max, min);

    using namespace boost::math::tools;
    typedef boost::math::tools::polynomial<double> Poly;
    eps_tolerance<double> eps_tol = eps_tolerance<double>(30);
    boost::uintmax_t max_iter = 50;

    Poly p(coeffs.begin(), coeffs.end());

    auto f = [&p](double x) { return p.evaluate(x); };
    if (max < min){
        double temp = max;
        max = min;
        min = temp;
    }

    const double minStep = (max - min) / 1e6;  // Smallest step allowed
    double left;
    double right;
    while (step >= minStep) {
        left = min;
        right = min + step;
        while (right <= max) {
            if (f(left) * f(right) <= 0) {
                auto result = toms748_solve(f, left, right, eps_tol,max_iter);
                return (result.first + result.second) / 2.0;
            }
            left = right;
            right += step;
        }
        step /= 10.0;
    }
    return std::numeric_limits<double>::quiet_NaN(); // No root found
}

template<int dim> 
void Ellipse<dim>::make_real_part_positive(std::complex<double>& z) {
    if (z.real() < 0) {
        z = std::complex<double>(-z.real(), z.imag());
    }
}
/// 
/// 
template<int dim> 
double Ellipse<dim>::radius_ellipse_between_foci(const std::complex<double>& p) {
    /*
        Continuous extension of the radius if p lies between the focal points   
   */
    std::complex<double> radius_bd = std::abs(eccentricity) / (-center + sqrt(center * center - eccentricity * eccentricity));
    std::complex<double> result = radius_bd * ((std::abs(p - center)) / std::abs(eccentricity)) ;
    return std::abs(result);
}
template<int dim> 
double Ellipse<dim>::radius_ellipse_complex(const std::complex<double>& p) {
    std::complex<double> diff_p = p - center;
    make_real_part_positive(diff_p); // cause of the complex sqrt
    std::complex<double> diff = -center;

    std::complex<double> scale = diff + std::sqrt(diff * diff - eccentricity * eccentricity);
    std::complex<double> numerator = diff_p + std::sqrt(diff_p * diff_p - eccentricity * eccentricity);
    
    return std::abs(numerator / scale);
}
template<int dim> 
double Ellipse<dim>::radius_p(const std::complex<double>& p) {
    // see Manteuffel " The Tchebychev Iteration for Nonsymmetric Linear Systems. "
    bool is_imag_zero = std::abs(p.imag()) < 1e-10;
    bool lies_above_the_center = std::abs(p.real() - center) < 1e-10;
    bool is_eccentricity_real_zero = std::abs(eccentricity.real()) < 1e-10;

    if (((is_imag_zero && std::abs(p - center) < std::abs(eccentricity)) && !is_eccentricity_real_zero )||
        (lies_above_the_center && (p.imag() < std::abs(eccentricity)) && is_eccentricity_real_zero)) {
        return radius_ellipse_between_foci(p); // fallback, point is between the foci
    } else {
        return radius_ellipse_complex(p);
    }
}

template<int dim> 
double Ellipse<dim>::radius_p_alterative(const std::complex<double>& p) {
    double px = p.real();
    double py = p.imag();

    px-= center;

    if (radiusx < 1e-13)
    {
        if (std::abs(px) < 1e-10){
            return py*py/(radiusy*radiusy);
        }
        else
            return 2.; // px is too small, return max value
    }
    if (radiusy < 1e-13)
    {
        if (std::abs(py) < 1e-10){
            return px*px/(radiusx*radiusx);
        }
        else
            return 2.; // p< is too small, return max value
    }

    return px*px/(radiusx*radiusx) + py*py/(radiusy*radiusy) ;

}

template<int dim>
 bool Ellipse<dim>::get_outer_points( std::vector<std::complex<double>>& outerpoints, const std::vector<std::complex<double>>& points){
    /*
        Identify points p whose radius(p) > radius(boundary_point) and sort them in descending order
    */
    outerpoints.clear();
    for (const auto& p : points) {
        // std::cout << "Point: " << p << " Radius: " << radius_p(p) - radius_ellipse  << " Radius_alterative: " << radius_p_alterative(p) ; ;
        //if (radius_p(p) > radius_ellipse + 1e-10) {
        if (radius_p_alterative(p) > 1. + 1e-10){
            outerpoints.push_back(p);
        }
    }

    if (!outerpoints.empty()) {
        std::sort(outerpoints.begin(), outerpoints.end(),
            [&](const std::complex<double>& a, const std::complex<double>& b) {
                return radius_p_alterative(a) > radius_p_alterative(b);  
            });
        return true;
    }
    return false;
 }

/////
/////
template <int dim> 
std::vector<int> Ellipse<dim>::create_opt_ellipse_mixed( std::vector<std::complex<double>>&  points){
    /*
    This function determines the optimal (smallest possible) ellipse that encloses a given set points (represented as complex numbers), using either 1 or 2 or 3 of the points as boundary-defining elements.
    */
    // int i = -1;
    double limit = 1; // to avoid degenerate ellipses
    // for (const auto& p : points) {
    //     i++;
    //     std::cout << "Point " <<  i << " : " << p ;
    // }   
    bool exists_outerpoints;
    double radius_min = std::numeric_limits<double>::max();
    double vol_min = std::numeric_limits<double>::max();
    std::vector<std::complex<double>> outerpoints;
    std::vector<int> best_config{-1,-1,-1};
    int number_points;
    if (points.size() == 1) {
        best_config[0] = 0;
        create_onepoint_ellipse(points[0]);
        return best_config;
    }
    else if(points.size() == 2){
        number_points = create_twopoint_ellipse_smallest_volume(points[0], points[1]);
        if(center + radiusx > limit) {
            // std::cout << "Found ellipse with two points: " << 0 << points[0] << " and "<< 1 << points[1] ;
            // if the center is too far left, shift the ellipse to the right and calc with the other function
            number_points = create_twopoint_ellipse(points[0],points[1], -1*limit);
        }
        // std::cout << "Radius of the opt ellipse:" << radius_ellipse << radius_p(points[0]) << radius_p(points[1]);
        if (number_points == 1) {
            if (std::abs(points[0].imag()) < std::abs(points[1].imag())) {
                best_config[0] = 1;
            } else {
                best_config[0] = 0;
            }
        } else if (number_points == 2) {
            best_config[0] = 0;
            best_config[1] = 1;
        } 
        return best_config;
    }
    // === Try all pairs of points ===
    // Try all point pairs to define an ellipse that contains all other points with minimum volume and is not to far in the right half plane
    bool too_far = false;
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            // std::cout << "Trying points: " << i << points[i] << " and "<< j << points[j] ;
            number_points = create_twopoint_ellipse_smallest_volume(points[i], points[j]);
            // std::cout << "Radius of the ellipse:" << radius_ellipse;
            too_far = false;
            if(center + radiusx > limit) {
                too_far = true;
            } 
            exists_outerpoints = get_outer_points(outerpoints,points);
            if(!exists_outerpoints && !too_far) {
                // std::cout << "Found ellipse with two points: " << i << points[i] << " and "<< j << points[j] ;
                if (number_points == 1){
                    if (std::abs(points[i].imag())  < std::abs(points[j].imag()))
                        best_config[0] = j;
                    else
                        best_config[0] = i;
                }
                else if (number_points == 2){
                    best_config[0] = i;
                    best_config[1] = j;
                }
                return best_config;
            }
        }
    }
    // === Try all pairs of points ===
    // Try all point pairs to define an ellipse that contains all other points minimize radius
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            // std::cout << "Trying points: " << i << points[i] << " and "<< j << points[j] ;
            number_points = create_twopoint_ellipse(points[i], points[j],-1*limit);
            // std::cout << "Radius of the ellipse:" << radius_ellipse;
            exists_outerpoints = get_outer_points(outerpoints,points);
            if(!exists_outerpoints) {
                // std::cout << "Found ellipse with two points: " << i << points[i] << " and "<< j << points[j] ;
                if (number_points == 1){
                    if (std::abs(points[i].imag())  < std::abs(points[j].imag()))
                        best_config[0] = j;
                    else
                        best_config[0] = i;
                }
                else if (number_points == 2){
                    best_config[0] = i;
                    best_config[1] = j;
                }
                return best_config;
            }
        }
    }
    // === Try all triplets of points ===
    //  If no such pair ellipse is valid, try all point triplets and keep the smallest.
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            for (size_t k = j + 1; k < points.size(); ++k) {
                number_points = create_threepoint_ellipse(points[i], points[j], points[k]);
                exists_outerpoints = get_outer_points(outerpoints,points);
                if(number_points != 0 && !exists_outerpoints && volume <  vol_min && center + radiusx < limit){
                    // std::cout << "Found ellipse with three points: " << i << points[i] << " and "<< j << points[j] << " and " << k << points[k] << std::endl;;
                    best_config[0] = i; 
                    best_config[1] = j; 
                    best_config[2] = k;
                    vol_min    = volume;
                }
                // std::cout << "Try points: " << i << points[i] << " and "<< j << points[j] << " and " << k << points[k] ;

            }
        }
    }
    if (best_config[0] > -1){
        create_threepoint_ellipse(points[best_config[0]],points[best_config[1]],points[best_config[2]]);
        return best_config;
    }

    // === Try all triplets of points ===
    //  If no such pair ellipse is valid, try all point triplets and keep the smallest.
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            for (size_t k = j + 1; k < points.size(); ++k) {
                number_points = create_threepoint_ellipse(points[i], points[j], points[k]);
                exists_outerpoints = get_outer_points(outerpoints,points);
                if(number_points != 0 && !exists_outerpoints && radius_ellipse <  radius_min){
                    // std::cout << "Found ellipse with three points: " << i << points[i] << " and "<< j << points[j] << " and " << k << points[k] ;
                    best_config[0] = i; 
                    best_config[1] = j; 
                    best_config[2] = k;
                    radius_min = radius_ellipse;
                }
                // std::cout << "Try points: " << i << points[i] << " and "<< j << points[j] << " and " << k << points[k] ;

            }
        }
    }
    if (best_config[0] > -1){
        create_threepoint_ellipse(points[best_config[0]],points[best_config[1]],points[best_config[2]]);
        return best_config;
    }
    // Return default (-1,-1,-1) if no valid ellipse was found
    return best_config;   
} 

template <int dim> 
std::vector<int> Ellipse<dim>::create_opt_ellipse( std::vector<std::complex<double>>&  points){
    /*
    This function determines the optimal (smallest possible) ellipse that encloses a given set points (represented as complex numbers), using either 1 or 2 or 3 of the points as boundary-defining elements.
    */
    // int i = -1;
    double limit = 1; // to avoid degenerate ellipses
    // for (const auto& p : points) {
    //     i++;
    //     std::cout << "Point " <<  i << " : " << p ;
    // }   
    bool exists_outerpoints;
    double radius_min = std::numeric_limits<double>::max();
    double vol_min = std::numeric_limits<double>::max();
    std::vector<std::complex<double>> outerpoints;
    std::vector<int> best_config{-1,-1,-1};
    int number_points;
    if (points.size() == 1) {
        best_config[0] = 0;
        create_onepoint_ellipse(points[0]);
        return best_config;
    }
    else if(points.size() == 2){
        number_points = create_twopoint_ellipse(points[0],points[1], -1*limit);

        // std::cout << "Radius of the opt ellipse:" << radius_ellipse << radius_p(points[0]) << radius_p(points[1]);
        if (number_points == 1) {
            if (std::abs(points[0].imag()) < std::abs(points[1].imag())) {
                best_config[0] = 1;
            } else {
                best_config[0] = 0;
            }
        } else if (number_points == 2) {
            best_config[0] = 0;
            best_config[1] = 1;
        } 
        return best_config;
    }
    // === Try all pairs of points ===
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            // std::cout << "Trying points: " << i << points[i] << " and "<< j << points[j] << std::endl;
            number_points = create_twopoint_ellipse(points[i], points[j],-1*limit);
            // std::cout << "Radius of the ellipse:" << radius_ellipse;
            exists_outerpoints = get_outer_points(outerpoints,points);
            if(!exists_outerpoints) {
                std::cout << "Found ellipse with two points: " << i << points[i] << " and "<< j << points[j] << std::endl;
                if (number_points == 1){
                    if (std::abs(points[i].imag())  < std::abs(points[j].imag()))
                        best_config[0] = j;
                    else
                        best_config[0] = i;
                }
                else if (number_points == 2){
                    best_config[0] = i;
                    best_config[1] = j;
                }
                return best_config;
            }
        }
    }
    // === Try all triplets of points ===

    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            for (size_t k = j + 1; k < points.size(); ++k) {
                number_points = create_threepoint_ellipse(points[i], points[j], points[k]);
                exists_outerpoints = get_outer_points(outerpoints,points);
                center -= limit;
                radius_ellipse = radius_p(points[i]-limit*std::complex<double>(1.,0));
                if(number_points != 0 && !exists_outerpoints && radius_ellipse <  radius_min){
                    std::cout << "Found ellipse with three points: " << i << points[i] << " and "<< j << points[j] << " and " << k << points[k] << std::endl ;
                    best_config[0] = i; 
                    best_config[1] = j; 
                    best_config[2] = k;
                    radius_min = radius_ellipse;
                }
                // std::cout << "Try points: " << i << points[i] << " and "<< j << points[j] << " and " << k << points[k] ;
            }
        }
    }
    center += limit;
    if (best_config[0] > -1){
        create_threepoint_ellipse(points[best_config[0]],points[best_config[1]],points[best_config[2]]);
        return best_config;
    }
    // Return default (-1,-1,-1) if no valid ellipse was found
    return best_config;   
} 

template<int dim> 
int Ellipse<dim>::create_onepoint_ellipse( std::complex<double> point){
    if (std::abs(point.real()) < 1e-10) {
        center = 0.0; // If the point is close to zero, set center to zero
    }
    else {
        center = point.real();
    } 
    radiusx = 0.;
    radiusy = std::abs(point.imag());

    eccentricity = 1i*std::abs(point.imag());
    if (radiusy < 1e-8){
        radius_ellipse = 0.;
    }
    else{
        radius_ellipse = radius_p(point);
    }
    return 1;
}

template<int dim> 
int Ellipse<dim>::create_twopoint_ellipse( std::complex<double> point1,  std::complex<double> point2, double shift ){
    double tol = 1e-8;
    // std::cout << "Creating two-point ellipse with points: " << point1 << " and " << point2 ;
    if (point1.imag() < 0) point1 = std::conj(point1);
    if (point2.imag() < 0) point2 = std::conj(point2);


    if (std::abs(point1.real() - point2.real()) < tol){
        std::complex<double> point_max_imag = std::complex<double> ( point1.real() , std::max(std::abs(point1.imag()), std::abs(point2.imag())) );
        return create_onepoint_ellipse(point_max_imag);
    }

    if (point1.real() * point2.real() < 0) {
        shift -=  std::min(std::abs(point1.real()), std::abs(point2.real())) + tol;
    }
    point1 = std::complex<double>(point1.real() + shift, point1.imag());
    point2 = std::complex<double>(point2.real() + shift, point2.imag());
    
    bool negative = false;
    double A = (point2.real() - point1.real()) / 2.0;
    double B = (point1.real() + point2.real()) / 2.0;
    double S = (point2.imag() - point1.imag()) / 2.0;
    double T = (point1.imag() + point2.imag()) / 2.0;
    //
    if (A < 0 || B < 0) {
        if (B < 0) {
            A *= -1;
            B *= -1;
            negative = true;
        }
        if (A < 0) {
            A *= -1;
            S *= -1;
        }
    }
    //
    if(std::abs(S) < tol){
        center = B* (negative ? -1 : 1);
        if(std::abs(T) < tol){
            radiusx = std::abs(center  - point1.real());
            eccentricity = radiusx;
            radiusy = 0;
        } else{
            double q0 = -A * A * A * A * B * B * (A * A + T * T);
            double q1 = 3 * A * A * A * A * B * B;
            double q2 = -3 * A * A * B * B;
            double q3 = B * B + T * T;
            double radius_x_2 = findRealRootBoost({q0, q1, q2, q3},B*B,A*A,0.1*A);
            std::complex<double> eccentricity_2 = radius_x_2 * (radius_x_2 - (A * A + T * T)) / (radius_x_2 - A * A);
            radiusx = std::sqrt(std::abs(radius_x_2));
            if (std::abs(eccentricity_2) < 1e-10){
                eccentricity_2 += 1e-9; // to avoid numerical issues
            }
            eccentricity = std::sqrt(eccentricity_2);
            radiusy = std::sqrt(std::abs(radius_x_2 - eccentricity_2));
        }
        center -= shift;
        point1 = std::complex<double>(point1.real() - shift, point1.imag());
        radius_ellipse = radius_p(point1);
        return 2;
        //
    } else if (std::abs(A) < tol) {
        center = B* (negative ? -1 : 1) ;
        std::complex<double> point_max = point1.imag() > point2.imag() ? point1 : point2;
        radiusx = 0;
        radiusy = point_max.imag();
        eccentricity = 1i*radiusy;
        center -= shift;
        point1 = std::complex<double>(point1.real() - shift, point1.imag());
        radius_ellipse = radius_p(point_max);
        return 2;
        //S != 0 and A != 0
    } else {
        double term = (T / S + S / T);
        double q0 = -3 * pow(A, 3) * S * T * (B * B - A * A);
        double q1 = -3 * pow(A, 3) * S * T * (2 * B - A * term);
        double q2 = A * S * T * ((B - A * T / S) * (B - 3 * A * T / S) +
        (B - A * S / T) * (B - 3 * A * S / T));
        double q3 = 4 * pow(A, 4) - 4 * pow(A, 3) * B * term + A * A * S * T * 
        ((pow(T, 3) / pow(S, 3) + pow(S, 3) / pow(T, 3)) - 3 * term) +
        A * A * B * B * (T * T / (S * S) + S * S / (T * T) + 3);
        double q4 = (2 * B + S * T / A - A * term) * ((2 * A * B + S * T) * term + 4 * A * A) +
        B * B * (2 * B - A * term) + B * (B * B - A * A);
        
        double q5 = (2 * B - A * term) * (2 * B + S * T / A - A * term);
        double r;
        // Find the real root of the polynomial
        if(S > 0){
            r = findRealRootBoost({q0, q1, q2, q3, q4, q5},A,0,0.1*A); 
        }
        else{
            r = findRealRootBoost({q0, q1, q2, q3, q4, q5},0,-A,0.1*A); 
        }
        if (std::isnan(r) || r == 0) {
            // std::cout << "FALL" << std::endl;
            r = (S > 0) ? A/2 : -A/2 ; // Fallback to A if no root found
        }
        // std::cout << "r: " << r << " A: " << A << " B: "<< B ;   
        center       = r + B ;
        radiusx      = std::sqrt((center - (B - A * T / S)) * (center - (B - A * S / T)));
        std::complex<double> eccentricity_2 = (  (center - (B + S * T / A)) *
                                    (center - (B - A * T / S)) *
                                    (center - (B - S * A / T))) / (center - B);
        if (std::abs(eccentricity_2) < 1e-10){
                eccentricity_2 += 1e-9; // to avoid numerical issues
            }
        eccentricity = std::sqrt(eccentricity_2);
           
        radiusy = std::sqrt(std::abs(radiusx*radiusx - eccentricity*eccentricity));
        center = center * (negative ? -1 : 1);
        center -= shift;
        point1 = std::complex<double>(point1.real() - shift, point1.imag());
        radius_ellipse = radius_p(point1);
        return 2;
    }
}

template<int dim> 
int Ellipse<dim>::create_twopoint_ellipse_smallest_volume( std::complex<double> point1,  std::complex<double> point2, double shift){
    double tol = 1e-8;
    // std::cout << "Creating two-point ellipse with points: " << point1 << " and " << point2 ;
    if (point1.imag() < 0) point1 = std::conj(point1);
    if (point2.imag() < 0) point2 = std::conj(point2);


    if (std::abs(point1.real() - point2.real()) < tol){
        std::complex<double> point_max_imag = std::complex<double> ( point1.real() , std::max(std::abs(point1.imag()), std::abs(point2.imag())) );
        return create_onepoint_ellipse(point_max_imag);
    }

    if (point1.real() * point2.real() < 0) {
        shift -=  std::min(std::abs(point1.real()), std::abs(point2.real())) + tol;
    }
    point1 = std::complex<double>(point1.real() + shift, point1.imag());
    point2 = std::complex<double>(point2.real() + shift, point2.imag());
    
    bool negative = false;
    double A = (point2.real() - point1.real()) / 2.0;
    double B = (point1.real() + point2.real()) / 2.0;
    double S = (point2.imag() - point1.imag()) / 2.0;
    double T = (point1.imag() + point2.imag()) / 2.0;
    //
    if (A < 0 || B < 0) {
        if (B < 0) {
            A *= -1;
            B *= -1;
            negative = true;
        }
        if (A < 0) {
            A *= -1;
            S *= -1;
        }
    }
    //
    if(std::abs(S) < tol){
        center = B* (negative ? -1 : 1);
        if(std::abs(T) < tol){
            radiusx = std::abs(center  - point1.real());
            eccentricity = radiusx;
            radiusy = 0;
        } else{
            double radius_x_2 = 2*A*A;
            std::complex<double> eccentricity_2 = radius_x_2 * (radius_x_2 - (A * A + T * T)) / (radius_x_2 - A * A);
            radiusx = std::sqrt(std::abs(radius_x_2));
            eccentricity = std::sqrt(eccentricity_2);
            radiusy = std::sqrt(std::abs(radius_x_2 - eccentricity_2));
        }
        center -= shift;
        point1 = std::complex<double>(point1.real() - shift, point1.imag());
        radius_ellipse = radius_p(point1);
        return 2;
        //
    } else if (std::abs(A) < tol) {
        center = B* (negative ? -1 : 1) ;
        std::complex<double> point_max = point1.imag() > point2.imag() ? point1 : point2;
        radiusx = 0;
        radiusy = point_max.imag();
        eccentricity = 1i*radiusy;
        center -= shift;
        point1 = std::complex<double>(point1.real() - shift, point1.imag());
        radius_ellipse = radius_p(point_max);
        return 2;
        //S != 0 and A != 0
    } else {
        double numerator = S*S + T*T  - std::sqrt(S*S*S*S + 14*S*S*T*T + T*T*T*T);
        center       = B -A*numerator / (6*S*T);
        radiusx      = std::sqrt((center - (B - A * T / S)) * (center - (B - A * S / T)));
        std::complex<double> eccentricity_2 = (  (center - (B + S * T / A)) *
                                    (center - (B - A * T / S)) *
                                    (center - (B - S * A / T))) / (center - B);
        eccentricity = std::sqrt(eccentricity_2);
           
        radiusy = std::sqrt(std::abs(radiusx*radiusx - eccentricity*eccentricity));
        center = center * (negative ? -1 : 1);
        center -= shift;
        point1 = std::complex<double>(point1.real() - shift, point1.imag());
        radius_ellipse = radius_p(point1);
        return 2;
    } 

}  

template<int dim> 
int Ellipse<dim>::create_twopoint_ellipse_smallest_volume_alternativ( std::complex<double> point1,  std::complex<double> point2){
    double tol = 1e-8;

    if (point1.imag() < 0) point1 = std::conj(point1);
    if (point2.imag() < 0) point2 = std::conj(point2);

    if (std::abs(point1.real() - point2.real()) < tol){
        std::complex<double> point_max_imag = std::complex<double> ( point1.real() , std::max(std::abs(point1.imag()), std::abs(point2.imag())) );
        return create_onepoint_ellipse(point_max_imag);
    }

    double x0, y0, x1, y1;
    if (point1.real() < point2.real()) {
        x0 = point1.real();
        y0 = point1.imag();
        x1 = point2.real();
        y1 = point2.imag();
    } else {
        x0 = point2.real();
        y0 = point2.imag();
        x1 = point1.real();
        y1 = point1.imag();
    }

    if (std::abs(x0 - x1) < tol) {
        center = x0;
        radiusy = std::max(y0, y1);
        radiusx = 0.;
        // eccentricity = std::complex<double>(0, radiusy);

    }
    else if (std::abs(y0 - y1) < tol){
        center = (x0 + x1) / 2.0;
        if (std::abs(y0) < tol) {
            radiusy = 0;
            radiusx = std::abs((x0 - x1) / 2.0);
        } else {
            radiusx = std::abs(std::sqrt(2.0) * (x1 - x0) / 2.0);
            double temp = 4 * radiusx * radiusx - x1 * x1 + 2 * x1 * x0 - x0 * x0;
            radiusy = 2 * radiusx * y0 / std::sqrt(temp);
        }
    }
    else{
        double numerator = y1 * y1 * y1 * y1 - 4 * y1 * y1 * y0 * y0 + y0 * y0 * y0 * y0;
        double root_term = pow(y1, 8) + pow(y1, 6) * y0 * y0 + y1 * y1 * pow(y0, 6) + pow(y0, 8);
        double denominator = y1 * y1 * y1 * y1 - 2 * y1 * y1 * y0 * y0 + y0 * y0 * y0 * y0;

        radiusx = std::abs(sqrt(2.0) * std::sqrt(numerator + std::sqrt(root_term)) / std::sqrt(denominator) * (x1 - x0) / 3.0);

        double b_sqrt_term = radiusx * radiusx * pow(y1, 4) - 2 * radiusx * radiusx * y1 * y1 * y0 * y0 + radiusx * radiusx * pow(y0, 4) + x1 * x1 * y1 * y1 * y0 * y0 - 2 * x1 * y1 * y1 * x0 * y0 * y0 + y1 * y1 * x0 * x0 * y0 * y0;

        double inner_term = x1 * y1 * y1 + x1 * y0 * y0 - y1 * y1 * x0 - x0 * y0 * y0 + 2 * std::sqrt(b_sqrt_term);

        double outer_term = 4 * radiusx * radiusx * x1 - 4 * radiusx * radiusx * x0 - pow(x1, 3) + 3 * x1 * x1 * x0 - 3 * x1 * x0 * x0 + x0 * x0 * x0;

        // if (std::abs(outer_term) < tol){
        //     center = x0;
        //     radiusy = std::max(y0, y1);
        //     radiusx = 0;
        //     eccentricity = std::complex<double>(0, radiusy);
        //     return 1;
        // } 
        {
            radiusy = std::abs(std::sqrt(inner_term) * radiusx / std::sqrt(outer_term));
            std::cout << "radiusy:" << radiusy << std::endl;
            std::cout << "out" << outer_term << std::endl;
    
            double denom = 2 * radiusy * radiusy * (x1 - x0);
            if (std::abs(denom) < tol) denom = tol;
    
            center = (y1 * y1 * radiusx * radiusx - y0 * y0 * radiusx * radiusx + radiusy * radiusy * x1 * x1 - radiusy * radiusy * x0 * x0) / denom;
        }

    }

    double focus_sq = radiusx * radiusx - radiusy*radiusy;

    if (focus_sq < 0)
        eccentricity = std::complex<double>(0, std::sqrt(-focus_sq));
    else
        eccentricity = std::sqrt(focus_sq);

    radius_ellipse = radius_p(point1);

    return 2;
}

template<int dim>
int Ellipse<dim>::create_threepoint_ellipse( std::complex<double> point1, std::complex<double> point2, std::complex<double> point3){
    double tol = 1e-8;
    std::vector<std::complex<double>> points = {point1, point2, point3};   
    sort(points.begin(), points.end(), [](const std::complex<double>& a, const std::complex<double>& b) {
        return std::real(a) < std::real(b);
    }); // using boost

    // if imag's are equal
    if (abs(imag(points[2]) - imag(points[0])) < tol &&
        abs(imag(points[1]) - imag(points[0])) < tol) {
        
        std::complex<double> point_min = points[0];
        std::complex<double> point_max = points[2];
        return create_twopoint_ellipse(point_max, point_min);
    } 
    // if one real is equal
    else if (abs(std::real(points[2]) - std::real(points[1])) < tol || abs(std::real(points[0]) - std::real(points[1])) < tol) {
        std::complex<double> point_max;
        if (abs(std::real(points[2]) - std::real(points[1])) < tol) {
            point_max = (imag(points[2]) > imag(points[1])) ? points[2] : points[1];
            return create_twopoint_ellipse(point_max, points[0]);
        } else {
            point_max = (imag(points[1]) > imag(points[0])) ? points[1] : points[0];
            return create_twopoint_ellipse(point_max, points[2]);
        }
    }
    // 
    else if ((std::real(points[1]) - std::real(points[0])) * (pow(imag(points[2]), 2) - pow(imag(points[0]), 2)) >= (std::real(points[2]) - std::real(points[0])) * (pow(imag(points[1]), 2) - pow(imag(points[0]), 2))) {
        return 0;
    }
    //
    else{
        double x0 = std::real(points[0]), x1 = std::real(points[1]), x2 = std::real(points[2]);
        double y0 = std::imag(points[0]), y1 = std::imag(points[1]), y2 = std::imag(points[2]);

        double den = y0*y0*(x1 - x2) + y1*y1*(x2 - x0) + y2*y2*(x0 - x1);
        if (std::abs(den) < tol) return 0;

        center = 0.5 * (y0*y0*(x1*x1 - x2*x2) + y1*y1*(x2*x2 - x0*x0) + y2*y2*(x0*x0 - x1*x1)) / den;

        double radiusx2 = center*center - (y0*y0*x1*x2*(x1 - x2) + y1*y1*x2*x0*(x2 - x0) + y2*y2*x0*x1*(x0 - x1)) / den;

        den = (x1 - x2)*(x2 - x0)*(x0 - x1);
        if (std::abs(den) < tol) return 0;

        std::complex<double> eccentricity_2 = radiusx2 * (1.0 - (y0*y0*(x1 - x2) + y1*y1*(x2 - x0) + y2*y2*(x0 - x1)) / den);
        if (std::abs(eccentricity_2) < 1e-10){
                eccentricity_2 += 1e-9; // to avoid numerical issues
            }
        radiusy = std::sqrt(std::abs(radiusx2 - eccentricity_2));
        radiusx = std::sqrt(radiusx2);
        eccentricity = std::sqrt(eccentricity_2);
        radius_ellipse = radius_p(point1);
        return 3; 
    } 
}


// RAII deleter for gsl_integration_workspace
inline auto make_gsl_workspace(std::size_t limit) {
    auto deleter = [](gsl_integration_workspace* w){ gsl_integration_workspace_free(w); };
    return std::unique_ptr<gsl_integration_workspace, decltype(deleter)>{
        gsl_integration_workspace_alloc(limit), deleter
    };
}

template<int dim>
double Ellipse<dim>::compute_l2_error_on_line(
    const std::function<std::complex<double>(std::complex<double>)>& f,
    const std::function<std::complex<double>(std::complex<double>)>& f_approx,
    double start,
    double end,
    std::size_t limit /*=1000*/ )
{
    gsl_set_error_handler_off();

    double length = std::abs(end - start);
    if (length == 0.0) return 0.0;

    // GSL integrand context
    struct Context {
        const std::function<std::complex<double>(std::complex<double>)>* f;
        const std::function<std::complex<double>(std::complex<double>)>* f_approx;
    } ctx{ &f, &f_approx };

    gsl_function F;
    F.function = [](double x, void* p) -> double {
        auto& c = *static_cast<Context*>(p);
        double diff = std::abs((*(c.f))(x) - (*(c.f_approx))(x));
        return diff * diff;
    };
    F.params = &ctx;

    auto W = make_gsl_workspace(limit);

    double result = 0.0, abserr = 0.0;
    int status = gsl_integration_qag(&F, start, end,
                                     1e-8, 1e-8,
                                     static_cast<int>(limit),
                                     GSL_INTEG_GAUSS61,
                                     W.get(),
                                     &result, &abserr);

    if (status != GSL_SUCCESS) {
        throw std::runtime_error(
            "[compute_l2_error_on_line] gsl_integration_qag failed with code " +
            std::to_string(status) + ", abserr=" + std::to_string(abserr));
    }

    return std::sqrt(result);
}

template<int dim>
double Ellipse<dim>::compute_l2_error_on_path(
    const std::function<std::complex<double>(std::complex<double>)>& f,
    const std::function<std::complex<double>(std::complex<double>)>& f_approx,
    const std::function<std::complex<double>(double)>& parametric_path,
    const std::function<std::complex<double>(double)>& parametric_path_deriv,
    double a, double b)
{
    gsl_set_error_handler_off(); // disable default error abort

    using IntegrandType = std::function<double(double)>;
    gsl_integration_workspace* w = gsl_integration_workspace_alloc(5000);

    IntegrandType integrand = [&](double phi) -> double {
        std::complex<double> z = parametric_path(phi);
        return std::norm(f(z) - f_approx(z)) * std::abs(parametric_path_deriv(phi));
    };

    gsl_function F;
    F.function = [](double phi, void* params) -> double {
        auto& integrand = *static_cast<IntegrandType*>(params);
        return integrand(phi);
    };
    F.params = &integrand;

    double result, abserr;
    int status = gsl_integration_qag(&F, a, b, 1e-8, 1e-8, 5000, GSL_INTEG_GAUSS51, w, &result, &abserr);
    if (status != GSL_SUCCESS) {
        std::cerr << "[compute_l2_error_on_path] gsl_integration_qag failed with code " +    std::to_string(status) + ", abserr=" + std::to_string(abserr) ;
        gsl_integration_workspace_free(w);
        return std::numeric_limits<double>::quiet_NaN();
    }
    gsl_integration_workspace_free(w);
    return std::sqrt(result);
}

template<int dim>
double Ellipse<dim>::compute_l_inf_error_on_path(
    const std::function<std::complex<double>(std::complex<double>)>& f,
    const std::function<std::complex<double>(std::complex<double>)>& f_approx,
    const std::function<std::complex<double>(double)>& parametric_path,
    double a, double b,
    std::size_t num_sample) // default 1000 sample points
{
    double max_error = 0.0;
    for (std::size_t i = 0; i <= num_sample; ++i) {
        double t = a + (b - a) * static_cast<double>(i) / static_cast<double>(num_sample);
        std::complex<double> z = parametric_path(t);
        double error = std::abs(f(z) - f_approx(z));
        if (error > max_error) {
            max_error = error;
        }
    }
    return max_error;
}

template<int dim>
double Ellipse<dim>::calc_error(
    std::function<std::complex<double>(std::complex<double>)> f,
    std::function<std::complex<double>(std::complex<double>)> f_approx,
    std::string norm)
{
    if (radiusy == 0) {
        const double start = center - radiusx;
        const double end   = center + radiusx;

        if (norm == "L2") {
            return compute_l2_error_on_line(f, f_approx, start, end);
        } else {
            return compute_l_inf_error_on_path(
                f, f_approx,
                [=](double x){ return center + radiusx * x; },
                -1., 1.);
        }
    } else if (radiusx == 0) {

        if (norm == "L2") {
            return compute_l2_error_on_path(f, f_approx,
                                            [=](double x){ return center + radiusy * x * 1i; },
                                            [=](double x){ return 0*x + radiusy  * 1i; },
                                            -1, 1);
        } else {
            return compute_l_inf_error_on_path(
                f, f_approx,
                [=](double x){ return center + radiusy * x * 1i; },
                -1., 1.);
        }
    } else {
        auto ellipse_param = [=](double phi) -> std::complex<double> {
            return center + radiusx * std::cos(phi)
                   + std::complex<double>(0.0, radiusy * std::sin(phi));
        };
        auto ellipse_param_deriv = [=](double phi) -> std::complex<double> {
            return -radiusx * std::sin(phi)
                   + std::complex<double>(0.0, radiusy * std::cos(phi));
        };
        if (norm == "L2")
            return compute_l2_error_on_path(f, f_approx,
                                            ellipse_param, ellipse_param_deriv,
                                            0., 2*M_PI);
        else
            return compute_l_inf_error_on_path(f, f_approx,
                                               ellipse_param, 0., 2*M_PI);
    }
}

template<int dim>
double Ellipse<dim>::calc_error_relative(
    std::function<std::complex<double>(std::complex<double>)> f,
    std::function<std::complex<double>(std::complex<double>)> f_approx,
    std::string norm) {

    // A null function for use with l2_error, always returns 0
    std::function<std::complex<double>(std::complex<double>)> null_func = [](std::complex<double>) {
    return std::complex<double>(0.0, 0.0);
    };
    double error = calc_error(f, f_approx,norm);
    double scale = calc_error(f, null_func,norm);
    
    if (scale < 1e-10) {
        return 0.0; // Avoid division by zero
    }
    
    return error / scale;    
}

template class Ellipse<1>;
template class Ellipse<2>;

}/* namespace Types */