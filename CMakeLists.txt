cmake_minimum_required(VERSION 3.13.4)
project(wave_semilinear LANGUAGES CXX )

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

find_package(deal.II 9.5.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
find_package(LAPACK REQUIRED COMPONENTS LAPACKE)

if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
endif()
DEAL_II_INITIALIZE_CACHED_VARIABLES()

include_directories(${LAPACKE_INCLUDE_DIRS})

enable_testing()

#include subdirectorys
add_subdirectory(processing)
add_subdirectory(data)
add_subdirectory(types)
add_subdirectory(generating)
add_subdirectory(time_integration)
add_subdirectory(tools)
add_subdirectory(algorithm)

# add examples
add_subdirectory(examples)

add_subdirectory(tests)

add_custom_target(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

add_custom_target(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

add_custom_target(run COMMAND wave_semilinear |tee ../output/LastRun.txt 
  COMMENT "Run with ${CMAKE_BUILD_TYPE} configuration"
  )

  # ignore BOOST deprecated headers
  add_definitions("-DBOOST_ALLOW_DEPRECATED_HEADERS")
  add_definitions("-DBOOST_BIND_GLOBAL_PLACEHOLDERS")



