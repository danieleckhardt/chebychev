#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/matrix_tools.h>

#include "../../types/Ellipse.h"

#include "../../time_integration/expoInt_tools/expoInt.h"
#include "../../time_integration/expoInt_tools/chebyshev_approx.h"
#include "../../tools/Output.h"

#include <iostream>
#include <complex>
#include <functional>


using namespace dealii;
using namespace MavesS;
using namespace Data;

using namespace std::complex_literals; // needs c++14

/*
    Tests to determine several different ellipses. 
*/
template<typename T>
double relative_error(T ref, T val)
{
    if (std::abs(ref) < 1e-10) {
        return std::abs(ref - val);
    } else {
        return std::abs(ref - val) / std::abs(ref);
    }
}

bool test_create_ellipse(Types::Ellipse<1>& ellipse,std::vector<std::complex<double>>& points, double a_ref, double b_ref, double center_ref, std::complex<double> foci_ref)
{
    std::vector<int> config(3);
    int numb_points = 0;
    if (points.size() == 1) {
        numb_points = ellipse.create_onepoint_ellipse(points[0]);
    } else if (points.size() == 2) {
        numb_points = ellipse.create_twopoint_ellipse(points[0], points[1]);
    } else if (points.size() == 3) {
        numb_points = ellipse.create_threepoint_ellipse(points[0], points[1], points[2]);
    } else {
        config = ellipse.create_opt_ellipse_mixed(points);
    }
    double a, b, center;
    std::complex<double> foci;
    ellipse.get_parameters(a, b, foci, center);
    double err = relative_error(a_ref,a) + relative_error(b_ref,b) + relative_error(center_ref,center) + relative_error(foci_ref,foci);
    if (err < 0.1 && !std::isnan(err)) {
        return true;
    } else {
        std::cerr << "Test failed for optimal ellipse: a=" << a << ", b=" << b 
                  << ", center=" << center << ", foci=" << foci 
                  << ", err="  << err << ". ";
        if(numb_points != 0){
            std::cerr << numb_points << " point(s) was/were used.";
        }
        else{
            std::cerr << " Points " << config[0] << ", " << config[1]<< " and " << config[2]<< " were used.";
        }
        return false;
    }
}

bool test_create_twopoint_ellipse_alternative(Types::Ellipse<1>& ellipse, std::complex<double> point1,std::complex<double> point2, double a_ref, double b_ref, double center_ref, std::complex<double> foci_ref)
{
    ellipse.create_twopoint_ellipse_smallest_volume(point1, point2);
    double a, b, center;
    std::complex<double> foci;
    ellipse.get_parameters(a, b, foci, center);
    double err = relative_error(a_ref,a) + relative_error(b_ref,b) + relative_error(center_ref,center) + relative_error(foci_ref,foci);
    if (err < 0.1 && !std::isnan(err)) {
        return true;
    } else {
        std::cerr << "Test failed for two-point (smalles volume) ellipse: a=" << a << ", b=" << b 
                  << ", center=" << center << ", foci=" << foci 
                  << ", err="  << err << ". ";
        return false;
    }
}
    

bool test_radius(Types::Ellipse<1>& ellipse, const std::complex<double>& point, double radius_ref)
{
    double radiusx, radiusy, center;
    std::complex<double> eccentricity;
    ellipse.get_parameters(radiusx, radiusy, eccentricity, center);

    double radius = ellipse.radius_p(point);
    double err = relative_error(radius_ref, radius);
    if (err < 0.1 && !std::isnan(err)) {
        return true;
    } else {
        std::cerr << "Test failed for radius: point=" << point << ", radius=" << radius 
                  << ", expected=" << radius_ref << ", err="  << err << ". ";
        return false;
    }
}

// bool test_estimate_degree(Types::Ellipse<1>& ellipse, std::function<std::complex<double>(std::complex<double>)> func, double tol, int max_iter, int degree_ref, int& degree)
// {
//     double temp = 1;  
//     TimeIntegration::Chebyshev<1> cheb_approx(5, ellipse, temp);
//     std::cout.setstate(std::ios_base::failbit);
//     cheb_approx.estimate_degree_ellipse_by_error(func, tol, max_iter);
//     std::cout.clear();
//     //cheb_approx.estimate_degree_ellipse_by_value_coef(func, tol, max_iter);
//     // cheb_approx.estimate_degree_ellipse_by_estimate(0, tol);
//     cheb_approx.get_degree(degree);
//     double err = std::abs(degree_ref - degree );
//     if ( err < 0.1 && !std::isnan(err)) {
//         return true;
//     } else {
//         std::cerr << "Test failed for estimated degree: degree=" << degree 
//                   << ", expected=" << degree_ref << ", err="  << err << ". ";
//         return false;
//     }
// }

// double l2_error_f(Types::Ellipse<1>& ellipse, std::function<std::complex<double>(std::complex<double>)> f, int degree)
// {   
//     double temp = 1;  
//     TimeIntegration::Chebyshev<1> cheb_approx(degree, ellipse, temp);
//     auto f_approx = [&](std::complex<double> z) {
//         return cheb_approx.scalar_run(f, z);
//     };
//     return ellipse.calc_error_relative(f, f_approx);
// }

int main()
{
    const unsigned int dim = 1;
    using namespace dealii;
    using namespace MavesS;

    double err;
    double err_relativ;
    bool   TEST_STILL_CORRECT = true;
    bool   TEST_SHAPED_CORRECT = true;
    bool   TEST_RADIUS_CORRECT = true;
    bool   TEST_DEGREE_CORRECT = true;
    //

    double a; double b; double center; std::complex<double> foci; int degree;
    std::string test_number;
    //
    double a_ref; double b_ref; double center_ref; std::complex<double> foci_ref; int degree_ref;
    double a_ref_2; double b_ref_2; double center_ref_2; std::complex<double> foci_ref_2; int degree_ref_2;
    //
    double radius_p1; double radius_p2; double radius_p3; double radius_p4;
    double radius_ref1; double radius_ref2; double radius_refref; double radius_refe;
    //
    std::complex<double> point1;std::complex<double> point2; std::complex<double> point3; std::complex<double> point4; std::complex<double> point5; 
    std::complex<double> point6;std::complex<double> point7; std::complex<double> point8; std::complex<double> point9; std::complex<double> point10; 
    std::complex<double> point11;std::complex<double> point12; std::complex<double> point13; std::complex<double> point14; std::complex<double> point15; 
    std::complex<double> point_radius1; std::complex<double> point_radius2; std::complex<double> point_radius3; std::complex<double> point_radiusref; std::complex<double> point_radiusrefe;std::complex<double> pointout1;std::complex<double> pointout2;std::complex<double> pointin;
    //
    std::vector<std::complex<double>> points;
    std::vector<int> bd_points;
    std::vector<int> config_ref;
    //
    Types::Ellipse<dim> ellipse;
    double temp = 1.;
    // TimeIntegration::Chebyshev<dim> cheb_approx(10, ellipse,temp);
    //
    // std::function<std::complex<double>(std::complex<double>)> func =  [](std::complex<double> z) {return std::exp(z);};
    // auto f_approx = [&](std::complex<double> z) {
    //     return cheb_approx.scalar_run(func, z);
    // };

    
    //
    // Example 1 - one point
    //
    a_ref = 0.; b_ref = 2; center_ref = -5.; foci_ref = 2i; degree_ref = 5; test_number = "1";
    points = { std::complex<double>(-5.,2.)};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number << " failed. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 2.1 - two equal points
    //
    a_ref = 0.; b_ref = 1; center_ref = -5.; foci_ref = 1i; degree_ref = 5; test_number = "2.1";
    points = { std::complex<double>(-5.,1.), std::complex<double>(-5.,1.) };
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    TEST_SHAPED_CORRECT = TEST_SHAPED_CORRECT && test_create_twopoint_ellipse_alternative(ellipse, points[0], points[1], a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    //err_relativ = l2_error_f(ellipse, func, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number << " failed. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 2.2 - two points with same imaginary part + radius test
    //
    a_ref = 3.409830383047529; b_ref = 2.1038144393470435; center_ref = -5.; foci_ref = 2.683450772037554; 
    a_ref_2 = 4.242640687119286; b_ref_2 = 1.414213562373095; center_ref_2 = -5.; foci_ref_2 = 4; 

    radius_ref1 = 0.7798064449807707; radius_ref2 = 0.42883555777329424; 
    radius_refref = 0.5980807485237131; radius_refe = 0.29108154370918293;
    degree_ref = 7; test_number = "2.2";
    //
    points = {std::complex<double>(-2.,1.),std::complex<double>(-8.,1.)};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    point_radius1 = std::complex<double>(-4.,3.);
    TEST_RADIUS_CORRECT = test_radius(ellipse, point_radius1, radius_ref1);
    point_radius2 = std::complex<double>(-4.,1.);
    TEST_RADIUS_CORRECT = TEST_RADIUS_CORRECT && test_radius(ellipse, point_radius2, radius_ref2);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    point_radiusref = std::complex<double>(center_ref + a_ref,0);
    TEST_RADIUS_CORRECT = TEST_RADIUS_CORRECT && test_radius(ellipse, point_radiusref, radius_refref);
    point_radiusrefe = std::complex<double>(center_ref ,0) + foci_ref;
    TEST_RADIUS_CORRECT = TEST_RADIUS_CORRECT && test_radius(ellipse, point_radiusrefe, radius_refe);
    TEST_SHAPED_CORRECT = TEST_SHAPED_CORRECT && test_create_twopoint_ellipse_alternative(ellipse, points[0], points[1], a_ref_2, b_ref_2, center_ref_2, foci_ref_2);
    if (!(TEST_SHAPED_CORRECT && TEST_RADIUS_CORRECT )) {
        std::cerr << "TEST " << test_number << " failed. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 2.3a - two points with same real part 
    //
    a_ref = 3049.67; b_ref = 0.; center_ref = -3049.67; foci_ref = 3049.67; degree_ref = 181;
    test_number = "2.3a";
    points = {std::complex<double>(-6099.34,0.), std::complex<double>(-4.19601e-07,0)};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 1000, degree_ref, degree);
    //err_relativ = l2_error_f(ellipse, func, degree);
    if (!(TEST_DEGREE_CORRECT && TEST_SHAPED_CORRECT)) {
        std::cerr << "TEST " << test_number << " failed. " ;
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 2.3b - two points with same real part 
    //
    a_ref = 0.; b_ref = 8.; center_ref = -4.; foci_ref = 8.*1i; degree_ref = 10 ; test_number = "2.3b"; 
    a_ref = 0.; b_ref = 8.; center_ref = -4.; foci_ref = 8.*1i;
    point1 = std::complex<double>(-4.,8.);
    point2 = std::complex<double>(-4.,2.);
    points = {point1, point2};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    TEST_SHAPED_CORRECT = TEST_SHAPED_CORRECT && test_create_twopoint_ellipse_alternative(ellipse, point1, point2, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number << " failed. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 2.4 - two points with p1.imag > p2.imag + radius tests
    //
    a_ref = 3.1993766593372626; b_ref = 9.851949841160117; 
    center_ref = -3.86724; foci_ref = 9.317988230536825*1i; 
    radius_ref1 = 0.214963269789061; radius_ref2 = 0.6752863415054146; 
    radius_refe = 0.7406217267094389; test_number = "2.4";
    degree_ref = 17;
    point1 = std::complex<double>(-2.,8.);
    point2 = std::complex<double>(-7.,2.);
    points = {point1, point2};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    ellipse.get_center(center);
    point_radius1 = std::complex<double>(center,3);
    TEST_RADIUS_CORRECT = test_radius(ellipse, point_radius1, radius_ref1);
    point_radius2 = std::complex<double>(center,0) + foci_ref;
    TEST_RADIUS_CORRECT = TEST_RADIUS_CORRECT && test_radius(ellipse, point_radius2, radius_ref2);
    point_radius3 = std::complex<double>(center,0) + foci_ref + 0.1;
    TEST_RADIUS_CORRECT = TEST_RADIUS_CORRECT && test_radius(ellipse, point_radius3, radius_refe);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT && TEST_RADIUS_CORRECT )) {
        std::cerr << "TEST " << test_number <<" FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 2.5a - two points with p1.imag < p2.imag + degree
    //
    a_ref = 5.132754600164406; b_ref = 9.277957352413623; center_ref = -6.368766244006702; foci_ref = 7.728862972501008*1i; test_number = "2.5a"; degree_ref = 17;
    a_ref_2 = 5.52112724823133; b_ref_2 = 8.496201897191662; center_ref_2 = -6.129036766322035; foci_ref_2 = 6.45775507329604*1i; 
    
    point1 = std::complex<double>(-11.,4.);
    point2 = std::complex<double>(-3.,7.);
    points = {point1, point2};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    TEST_SHAPED_CORRECT = TEST_SHAPED_CORRECT && test_create_twopoint_ellipse_alternative(ellipse, point1, point2, a_ref_2, b_ref_2, center_ref_2, foci_ref_2);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number <<" FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 2.5b - two points with p1.imag < p2.imag - switch points and negative imaginary part
    //
    a_ref = 5.132754600164406; b_ref = 9.277957352413623; center_ref = -6.368766244006702; foci_ref = 7.728862972501008*1i; degree_ref = 17; test_number = "2.5b";
    point1 = std::complex<double>(-3.,-7.);
    point2 = std::complex<double>(-11.,-4.);
    points = {point1, point2};
    TEST_SHAPED_CORRECT = test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT = test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number <<" FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 2.6 - two points - outer points
    //
    a_ref = 3.66686759611392; b_ref = 7.061598307700514; center_ref = -6.5167254185911965; foci_ref = 6.034919443695044*1i; degree_ref = 12; test_number = "2.6";
    point1 = std::complex<double>(-3.,2.);
    point2 = std::complex<double>(-7.,7.);
    points = {point1, point2};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points , a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number <<" FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    pointout1 = std::complex<double>(-3.,10.);
    pointout2 = std::complex<double>(-4.,10.);
    pointin = std::complex<double>(-6.5167254185911965,2.);
    points = {point2,pointout2,pointout1,pointin};
    std::vector<std::complex<double>> out_points = {};
    ellipse.get_outer_points(out_points,points);
    err = std::abs(pointout1 - out_points[0]) + std::abs(pointout2 - out_points[1]);
    if (err >= 0.1 || std::isnan(err)) {
        std::cerr << "Outer points do not match: expected " 
                  << pointout1 << " and " << pointout2 << ", got " 
                  << out_points[0] << " and " << out_points[1] 
                  << ". TEST " << test_number << " FAILED. " ;
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 2.7a - two points - big numbers
    //
    a_ref = 605.3250852791507; b_ref = 36.47862409953503; center_ref = -609.6749147208493; foci_ref = 604.2249323323359; degree_ref = 61; test_number = "2.7a";
    point1 = std::complex<double>(-1215.,0.);
    point2 = std::complex<double>(-8.,4.);
    points = {point1, point2};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);    
    if (!(TEST_SHAPED_CORRECT  )) {
        std::cerr << "TEST " << test_number <<"  FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 2.7b - two points with same real part 
    //
    a_ref = 3049.68; b_ref = 298.27; center_ref = -3049.68; foci_ref = 3035.06; degree_ref = 471;
    test_number = "2.7b";
    points = {std::complex<double>(-6099.34,1.), std::complex<double>(-4.19601e-07,0)};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 1000, degree_ref, degree);
    if (!(TEST_DEGREE_CORRECT && TEST_SHAPED_CORRECT)) {
        std::cerr << "TEST " << test_number << " failed. " ;
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 2.8 - two points - small positive numbers
    //
    a_ref =  4.025968094378676; b_ref = 24.98356688113653; center_ref = -4.025967094378677; foci_ref = 24.657051628434147*1i; degree_ref = 36; test_number = "2.8";
    point1 = std::complex<double>(1e-6,0.);
    point2 = std::complex<double>(-8.,4.);
    points = {point1, point2};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number <<"  FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 2.9 - two points - numbers where roots do not exist
    //
    a_ref =  1.0698600000000011; b_ref =  5.27381450612363; center_ref = -4.830139999999999; foci_ref = 5.164157145691833*1i;degree_ref = 8; test_number = "2.9";
    point1 = std::complex<double>(-5.9,0.);
    point2 = std::complex<double>(-4.47352,4.9722);
    points = {point1, point2};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number <<"  FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 2.10 - two points - extra
    //
    a_ref =  15.3203610757867; b_ref =  5.762219185302066; center_ref = -16.2905389242133; foci_ref = 14.19543213688886; degree_ref =17;  test_number = "2.10";
    point1 = std::complex<double>(-31.6109,0.);
    point2 = std::complex<double>(-2.4872,2.5);
    points = {point1, point2};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number <<"  FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 2.11 - two points - extra
    //
    a_ref =  153.783; b_ref =  152.408; center_ref = -156.828; foci_ref = 20.5151; degree_ref =179;  test_number = "2.11";
    point1 = std::complex<double>(-310.6109,0.);
    point2 = std::complex<double>(-20.4872,70.5);
    points = {point1, point2};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number <<"  FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 3.1 - three equal points 
    //
    a_ref = 0.; b_ref = 2; center_ref = -7.; foci_ref = 2i; test_number = "3.1";
    point1 = std::complex<double>(-7.,2.);
    point2 = std::complex<double>(-7.,2.);
    point3 = std::complex<double>(-7.,2.);
    points = {point1, point2, point3};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    if (!TEST_SHAPED_CORRECT) {
        std::cerr << "TEST " << test_number <<"  FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 3.2 - three points -two equal points 
    //
    a_ref = 2.8124235962667923; b_ref = 3.6890897732100463; center_ref = -4.63675216399285; foci_ref = 2.387395415502994*1i; test_number = "3.2";
    point1 = std::complex<double>(-7.,2.);
    point2 = std::complex<double>(-7.,2.);
    point3 = std::complex<double>(-3.,3.);
    points = {point1, point2, point3};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    if (!TEST_SHAPED_CORRECT) {
        std::cerr << "TEST " << test_number <<"  FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 3.3 - three points - real eccentricity
    //
    a_ref = 5.366563145999495; b_ref = 3.; center_ref = -6.; foci_ref = 4.449719092257398; degree_ref = 10;  test_number = "3.3";
    point1 = std::complex<double>(-10.,2.);
    point2 = std::complex<double>(-6.,3.);
    point3 = std::complex<double>(-2.,2.);
    points = {point1, point2, point3};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number <<" FAILED." ;
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 3.4 - three points - imaginary eccentricity
    //
    a_ref = 1.0795613245600981; b_ref = 6.010743070375289; center_ref = -2.935483870967742; foci_ref = 5.913000896717216*1i; degree_ref = 11; test_number = "3.4";
    point1 = std::complex<double>(-4.,1.);
    point2 = std::complex<double>(-3.,6.);
    point3 = std::complex<double>(-2.,3.);
    points = {point1, point2, point3};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number <<" FAILED." ;
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 4.0 
    //
    a_ref = 8.215150000000001 ; b_ref = 5.372687860557879; center_ref = -8.21515; foci_ref = 6.214733676957849; radius_ref1 =  1.;degree_ref = 15; test_number = "4.0";
    point1 = std::complex<double>(-16.4285,0);
    point2 = std::complex<double>(-13.5981,0);
    point3 = std::complex<double>(-2.784,-4.03105);
    point4 = std::complex<double>(-4.82953e-08,0);
    points = {point1, point2, point3, point4};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    TEST_RADIUS_CORRECT = test_radius(ellipse, point1, radius_ref1);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT && TEST_RADIUS_CORRECT )) {
        std::cerr << "TEST " << test_number <<" FAILED." ;
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 4.1a - more than three points - pretest
    //
    a_ref = 3.479225785294656; b_ref = 10.840640762040618; center_ref = -4.343347639484978; foci_ref = 10.2671554028639*1i; radius_ref1 =  0.9243750300444004;degree_ref = 18; test_number = "4.1a";
    point1 = std::complex<double>(-7.,7.);
    point2 = std::complex<double>(-3.,10.);
    point3 = std::complex<double>(-1.,3.);
    points = {point1, point2, point3};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    TEST_RADIUS_CORRECT = test_radius(ellipse, point1, radius_ref1);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT && TEST_RADIUS_CORRECT )) {
        std::cerr << "TEST " << test_number <<" FAILED. ";
        TEST_STILL_CORRECT = false;
    }

    //
    // Example 4.2a - more than three points- need three points
    //
    a_ref = 3.670822624309178; b_ref = 10.62321099951886; center_ref = -4.2388059701492535; foci_ref = 9.968835097500529*1i; degree_ref = 19; test_number = "4.2a";
    point1 = std::complex<double>(-1.,5.);
    point2 = std::complex<double>(-7.,7.);
    point3 = std::complex<double>(-5.,2.);
    point4 = std::complex<double>(-3.,10.);
    point5 = std::complex<double>(-1.,3.);
    points = {point1,point2,point3,point4,point5};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number << " FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 4.3 - more than three points - need two point
    //   
    a_ref = 4.01638; b_ref = 14.0026; center_ref = -4.129240511614394; foci_ref = 14.184379869769185*1i;degree_ref = 18; test_number = "4.3";
    point1 = std::complex<double>(-4.,3.);
    point2 = std::complex<double>(-4.,14.);
    point3 = std::complex<double>(-8.,3.);
    point4 = std::complex<double>(-1.,9.);
    point5 = std::complex<double>(-1.,5.);
    points = {point1,point2,point3,point4,point5};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number << " FAILED. ";
        TEST_STILL_CORRECT = false;
    }

    //
    // Example 4.4 - more than three points - need two point
    //   
    a_ref = 13.148320404465986; b_ref = 6.10637438526121; center_ref = -13.452136078953433; foci_ref = 11.644334301518587;degree_ref = 18;test_number = "4.4";
    point1 = std::complex<double>(-7.,2.);
    point2 = std::complex<double>(-21.,5.);
    point3 = std::complex<double>(-3.,1.);
    point4 = std::complex<double>(-5.,1.);
    point5 = std::complex<double>(-2.,3.);
    points = {point1,point2,point3,point4,point5};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number << " FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 4.5a - more than three points - big numbers
    //   
    a_ref = 600.4678926410625; b_ref = 85.16328404114094; center_ref = -599.5321073589375; foci_ref = 594.3979350099784; degree_ref = 169; test_number = "4.5a";
    point1 = std::complex<double>(-1200.,0.);
    point2 = std::complex<double>(-0.1,5.);
    point3 = std::complex<double>(-3.,1.);
    point4 = std::complex<double>(-5.,1.);
    point5 = std::complex<double>(-2.,3.);
    points = {point1,point2,point3,point4,point5};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number << " FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 4.5b - more than three points - big numbers
    //   
    a_ref = 50.46220933568413; b_ref = 24.94105260502241; center_ref = -49.53779066431587; foci_ref =43.8677383277496; degree_ref = 50;  test_number = "4.5b";
    point1 = std::complex<double>(-100.,0.);
    point2 = std::complex<double>(-0.1,5.);
    point3 = std::complex<double>(-3.,1.);
    point4 = std::complex<double>(-5.,1.);
    point5 = std::complex<double>(-2.,3.);
    points = {point1,point2,point3,point4,point5};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number << " FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 4.6 - more than three points 
    //   
    a_ref =  15.9112; b_ref =  5.22056; center_ref = -15.6997; foci_ref = 15.0304;  degree_ref = 18;  test_number = "4.6";
    point1 = std::complex<double>(-31.6109,0);
    point2 = std::complex<double>(-22.5394,0);
    point3 = std::complex<double>(-13.3379,0);
    point4 = std::complex<double>(-0.249927,1.08957);
    point5 = std::complex<double>(-0.249927,-1.08957);
    point6 = std::complex<double>(-1.03866,2.02842);
    point7 = std::complex<double>(-1.03866,-2.02842);
    point8 = std::complex<double>(-2.48721,2.49997);
    point9 = std::complex<double>(-2.48721,-2.49997);
    point10 = std::complex<double>(-4.8,0.979796);
    point11 = std::complex<double>(-4.8,-0.979796);
    point12= std::complex<double>(-3.07668,0);
    point13 = std::complex<double>(-2.81189,0);
    point14 = std::complex<double>(-2.7147,0);
    point15 = std::complex<double>(0,0);
    points = {point1,point2,point3,point4,point5,point6,point7,point8,point9,point10,point11,point12,point13,point14,point15};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number << " FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 4.7 - more than three points 
    //   
    a_ref =  0; b_ref =  2.49997; center_ref = 0; foci_ref = 2.49997*1i;  degree_ref = 7;  test_number = "4.7";
    point1 = std::complex<double>(0,0);
    point2 = std::complex<double>(0,0);
    point3 = std::complex<double>(0,0);
    point4 = std::complex<double>(0,1.08957);
    point5 = std::complex<double>(1e-16,-1.08957);
    point6 = std::complex<double>(0,2.02842);
    point7 = std::complex<double>(1e-17,-2.02842);
    point8 = std::complex<double>(1e-16,2.49997);
    point9 = std::complex<double>(1e-17,-2.49997);
    point10 = std::complex<double>(1e-17,0.979796);
    point11 = std::complex<double>(1e-17,-0.979796);
    point12= std::complex<double>(1e-15,0);
    point13 = std::complex<double>(1e-17,0);
    point14 = std::complex<double>(1e-17,0);
    point15 = std::complex<double>(0,0);
    points = {point1,point2,point3,point4,point5,point6,point7,point8,point9,point10,point11,point12,point13,point14,point15};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number << " FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 4.8 - more than three points 
    //   
    a_ref = 302.295; b_ref = 48.2067; center_ref = -301.937; foci_ref =298.427; degree_ref = 117;  test_number = "4.8";
    point1 = std::complex<double>(-604.232,0);
    point2 = std::complex<double> (-6.29333,9.28766);
    point3 = std::complex<double>(-2.23676,6.30334);
    point4 = std::complex<double>(-3.99883,7.99912);
    point5 = std::complex<double>(-0.990135,4.33847);
    point6 = std::complex<double>(-2.23676,6.30334);
    points = {point1,point2,point3,point4,point5,point6};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number << " FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 4.9 - more than three points 
    //   
    a_ref = 352.296; b_ref = 52.0102; center_ref = -351.936; foci_ref = 348.436; degree_ref = 121;  test_number = "4.9";
    point1 = std::complex<double>(-704.232,0);
    point2 = std::complex<double>(-6.29333,9.28766);
    point3 = std::complex<double>(-2.23676,6.30334);
    point4 = std::complex<double>(-3.99883,7.99912);
    point5 = std::complex<double>(-0.990135,4.33847);
    point6 = std::complex<double>(-2.23676,6.30334);
    points = {point1,point2,point3,point4,point5,point6};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 100, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number << " FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    // Example 4.10 - more than three points 
    //   
    a_ref = 2020.57; b_ref = 1069.66; center_ref = -2019.66; foci_ref = 1714.22; degree_ref = 1062;  test_number = "4.10";
    point1 = std::complex<double>(-4040.232,0);
    point2 = std::complex<double>(-6.29333,90.28766);
    point3 = std::complex<double>(-2.23676,6.30334);
    point4 = std::complex<double>(-3.99883,7.99912);
    point5 = std::complex<double>(-0.990135,4.33847);
    point6 = std::complex<double>(-2.23676,6.30334);
    points = {point1,point2,point3,point4,point5,point6};
    TEST_SHAPED_CORRECT =  test_create_ellipse(ellipse, points, a_ref, b_ref, center_ref, foci_ref);
    // TEST_DEGREE_CORRECT =  test_estimate_degree(ellipse, func, 0.01, 1000, degree_ref, degree);
    if (!(TEST_SHAPED_CORRECT )) {
        std::cerr << "TEST " << test_number << " FAILED. ";
        TEST_STILL_CORRECT = false;
    }
    //
    //
    //
    if(TEST_STILL_CORRECT == true){ 
        std::cout<< "1";
    }else{
        std::cout<< "End of tests, some FAILED. ";
    }
}

