find_package(PkgConfig REQUIRED)
find_package(Eigen3 REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${FFTW_INCLUDE_DIRS}
  /usr/local/include         # Spectra
)

ADD_LIBRARY(time_integration
  imp_Midpoint.cpp
  IMEXE.cpp
  IMEXR.cpp
  explicit_Midpoint.cpp
  IMEXArs3.cpp
  TimeIntegrator.cpp
  expoEuler.cpp
  expoInt_tools/expoInt.cpp
  expoInt_tools/chebyshev_approx.cpp
  expoInt_tools/calc_eigen.cpp
  expoInt_tools/arnoldikrylov.cpp
  expoInt_tools/findellipse.cpp
  )
  
TARGET_LINK_LIBRARIES(time_integration Eigen3::Eigen data tools types processing ${FFTW_LIBRARIES} )
DEAL_II_SETUP_TARGET(time_integration)