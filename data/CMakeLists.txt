INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

ADD_LIBRARY(data
  Boundary.cpp
  Coefficients.cpp
  ExactSolution.cpp
  InitialConditions.cpp
  RHS.cpp
  )

DEAL_II_SETUP_TARGET(data)