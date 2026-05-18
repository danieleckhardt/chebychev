set(LAPACK_LIBRARIES lapacke lapack blas) # Explicitly link all required libraries:
SET(TEST_LIBRARIES tools generating data time_integration processing types ${LAPACK_LIBRARIES} )
ADD_SUBDIRECTORY(conv)
ADD_SUBDIRECTORY(micro)
ADD_SUBDIRECTORY(matrix_func)
ADD_SUBDIRECTORY(types)
