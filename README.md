## Getting started
### Requirements

- [ ] CMake 3.13 or later
- [ ] dealii 9.6
- [ ] GCC 11.1

### Compilation

Enter the repository and create a build folder
```
cd chebychev/
mkdir build
cd build/
```
Next, configure the project 
```
 cmake ..
```
Build either in debug or release mode.
```
make debug
make release
```
### Example 
To get started quickly, you’ll find examples in the `examples` folder that contain the most important files for creating the data. Python files for creating the plots and tables are available in the `tools` folder.

### Using Docker 
If you want to use the docker file you have to install docker on you operating system (for Ubuntu see https://docs.docker.com/engine/install/ubuntu/).

The docker file is located in the folder `.dockercontainer`.

#### Setup in VS Code
With VS Code you can reopen the folder in a container and start compiling by using the Dev Containers extension.  For more details see [here](https://code.visualstudio.com/docs/devcontainers/containers).

#### Setup directly

```
docker build -t dealii .devcontainer/
```
Start a container
```
docker run -i -t -v .:/home/dealii/nlmaves dealii
```
Continue the compilation as described above.  ( Does not work on Network Boot devices of KIT! )

