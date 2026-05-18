# Example: Plots from EckhardtHochbruckVerfuerth24

This is the example, presenting in  from EckhardtGrimmHochbruck26. 

## Build and execute
After setting up the CMake project, enter at the root of the build folder the example folder
```bash
    cd examples/EckhardtGrimmHochbruck26/
```
To generate the data for spatial and temporal convergence plots, build the target `runall_Paper26` and execute the binary

```bash
    make 
    ./runall_Paper26
```
The data can also be generated separately by 

```bash
    cd chebyshev_arnoldi/
    make 
    ./chebyshev_arnoldi
```
or  
```bash
    cd matrix_functions_conv_velocity/
    make 
    ./matrix_functions_conv_velocity
```
or  
```bash
    cd matrix_functions_strong_damp/
    make 
    ./matrix_functions_strong_damp
```

## Output
The result can be found in the folder `output/errors/Example101` and `output/errors/Example201` and `output/errors/Example301`. There are three different subfolders `cheb` and `poly_krylov` and `rat_krylov`, in which the data for the plots are stored, respectively. There are Python scripts for the visualization, which are located in `tools/`. 

