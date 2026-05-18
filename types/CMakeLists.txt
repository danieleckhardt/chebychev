INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(
    /usr/include/petsc
    /usr/include/slepc
)

link_directories(
    /usr/lib
)

find_package(Boost REQUIRED)
find_package(GSL REQUIRED)

ADD_LIBRARY(types 
  InverseMatrix.cpp
  InverseMatrixPETSc.cpp
  Ellipse.cpp
  BlockMatrix.cpp
  BlockMatrixOp.cpp
  Functions_for_matrix.cpp
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(types PRIVATE
        -funroll-all-loops
        -Wno-placement-new
        -Wno-literal-suffix
    )
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(types PRIVATE
        -O3
        -Wno-literal-range
    )
endif()

TARGET_LINK_LIBRARIES(types data Boost::boost GSL::gsl petsc slepc)

DEAL_II_SETUP_TARGET(types)