INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

ADD_LIBRARY(generating
  ${CMAKE_CURRENT_SOURCE_DIR}/MicroProblem.cpp
  )
  
TARGET_LINK_LIBRARIES(generating data)
DEAL_II_SETUP_TARGET(generating)

  # ignore BOOST deprecated headers
  add_definitions("-DBOOST_ALLOW_DEPRECATED_HEADERS")
  add_definitions("-DBOOST_BIND_GLOBAL_PLACEHOLDERS")