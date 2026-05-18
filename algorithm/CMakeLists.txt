INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

ADD_LIBRARY(algorithm
  ${CMAKE_CURRENT_SOURCE_DIR}/global.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/wave_semilinear.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matrix_function.cpp

  )
find_package(Boost REQUIRED)

set(LAPACK_LIBRARIES lapacke lapack blas) # Explicitly link all required libraries:

TARGET_LINK_LIBRARIES(algorithm data generating processing time_integration tools ${LAPACK_LIBRARIES} Boost::boost )

DEAL_II_SETUP_TARGET(algorithm)