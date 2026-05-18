#ifndef DATA_TYPES_ELLIPSE_H_
#define DATA_TYPES_ELLIPSE_H_

#include<complex>
#include <cmath>
#include <memory>
#include <functional>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_errno.h>
#include <boost/math/tools/roots.hpp>
#include <boost/math/tools/polynomial.hpp>


using namespace std::complex_literals; // needs c++14

namespace Types{

template<int dim>
class Ellipse{
    public:
    Ellipse(double radiusx = 1., std::complex<double> eccentricity = 0., double center = 0.);
    Ellipse(Ellipse& ellipse) = default;
    Ellipse(const Ellipse& ellipse) = default;
    Ellipse& operator=(const Ellipse& ellipse) = default;
    ~Ellipse() = default;
    //
    void get_parameters( double& radiusx, double& radiusy, std::complex<double>& eccentricity, double& center);
    void get_radx( double& radiusx);
    void get_rady( double& radiusy);
    void get_center( double& center);
    void get_ecc( std::complex<double>& eccentricity);
    void get_radius_ellipse(double& radius_ell);
    void get_volume_ellipse(double& radius_vol);
    //
    void set_parameters( double radiusx, double radiusy,  double center);
    void set_radx( double radiusx);
    bool set_radx_with_fix_center_and_ecc( double radiusx);
    bool set_rady_with_fix_center_and_ecc( double radiusy);
    void set_rady( double radiusy);
    void set_center( double center);
    void set_ecc( std::complex<double> eccentricity);
    //
    std::vector<int> create_opt_ellipse( std::vector<std::complex<double>>& points); 
    std::vector<int> create_opt_ellipse_mixed( std::vector<std::complex<double>>& points); 
    //
    int create_onepoint_ellipse(std::complex<double> point);
    int create_twopoint_ellipse( std::complex<double> point1,  std::complex<double> point2, double shift = 0.);
    int create_twopoint_ellipse_smallest_volume_alternativ( std::complex<double> point1,  std::complex<double> point2);
    int create_twopoint_ellipse_smallest_volume( std::complex<double> point1,  std::complex<double> point2,double shift = 0.);
    int create_threepoint_ellipse( std::complex<double> point1, std::complex<double> point2, std::complex<double> point3);
    //
    double radius_p(const std::complex<double>& p);
    double radius_p_alterative(const std::complex<double>& p);
    bool   get_outer_points( std::vector<std::complex<double>>& outerpoints, const std::vector<std::complex<double>>& points);
    //
    double calc_error_relative(std::function<std::complex<double>(std::complex<double>)> f,std::function<std::complex<double>(std::complex<double>)> f_approx, std::string norm = "L2");
    double calc_error(std::function<std::complex<double>(std::complex<double>)> f, std::function<std::complex<double>(std::complex<double>)> f_approx,std::string norm= "L2" );
    //
    //
    private:
        double radius_ellipse;
        double volume{};
        double radiusx{};
        double radiusy{};
        double center{}; // only x-coordinate, y-coordinate always 0
        std::complex<double> eccentricity{};

        double findRealRootBoost( const std::vector<double>& coeffs, double max , double min =  0., double step = 0.5);
        void   make_real_part_positive(std::complex<double>& z);
        double radius_ellipse_between_foci(const std::complex<double>& p);
        double radius_ellipse_complex(const std::complex<double>& p);
        double compute_l2_error_on_line(const std::function<std::complex<double>(std::complex<double>)>& f,const std::function<std::complex<double>(std::complex<double>)>& f_approx,double start,double end,std::size_t limit  =1000 );
        double compute_l2_error_on_path(const std::function<std::complex<double>(std::complex<double>)>& f,const std::function<std::complex<double>(std::complex<double>)>& f_approx, const std::function<std::complex<double>(double)>& parametric_path, const std::function<std::complex<double>(double)>& parametric_path_deriv, double a, double b);
        double compute_l_inf_error_on_path(const std::function<std::complex<double>(std::complex<double>)>& f, const std::function<std::complex<double>(std::complex<double>)>& f_approx, const std::function<std::complex<double>(double)>& parametric_path, double a, double b,  std::size_t num_sample = 1000);
};
}/* namespace Types */
# endif //DATA_TYPES_ELLIPSE_H_
