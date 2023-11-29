# General-Purpose Computation on RT Cores

<!--- ------------------------------------------------------- -->
Though the RT cores were built to accelerate ray tracing applications, we show that we can re-structure problems to resemble ray tracing problems that can leverage the hardware acceleration. 

In this repository, we implement a clustering application (DBSCAN) and k-Nearest Neighbor Search application that uses RT acceleration. Please follow the [installation instructions](https://github.com/vani-nag/OWLRayTracing#building-owl--supported-platforms) to install and configure OWL.

<!-- ### RT-DBSCAN 
RT-DBSCAN offloads distance computations in DBSCAN to the ray tracing cores and performs other clustering operations in shader cores. The implementation and execution instructions are in the [RT-DBSCAN](https://github.com/vani-nag/OWLRayTracing/tree/master/samples/s02-rtdbscan) directory. -->

### TrueKNN 
TrueKNN removes the fixed-radius constraint in existing translations of Nearest Neighbor queries to Ray Tracing queries. The implementation and execution instructions are in the [TrueKNN](https://github.com/vani-nag/OWLRayTracing/tree/master/samples/s01-trueknn) directory.

If you find this work to be useful in your research, please cite:
1. V. Nagarajan, D. Mandarapu, and M. Kulkarni.  "RT-kNNS Unbound: Using RT Cores to Accelerate Unrestricted Neighbor Search". In Proceedings of the 37th International Conference on Supercomputing (ICS '23). Association for Computing Machinery, New York, NY, USA, 289–300
2. V. Nagarajan and M. Kulkarni, "RT-DBSCAN: Accelerating DBSCAN using Ray Tracing Hardware," 2023 IEEE International Parallel and Distributed Processing Symposium (IPDPS), St. Petersburg, FL, USA, 2023, pp. 963-973

<!--- ------------------------------------------------------- -->
# Building OWL / Supported Platforms

General Requirements:
- OptiX 7 SDK (version 7.0, 7.1, or 7.2, will work with either)
- CUDA verion 10 or 11
- a C++11 capable compiler (regular gcc on CentOS and Linux should do, VS on Windows)

Per-OS Instructions:

- Ubuntu 18, 19, and 20 (automatically tested on 18, mostly developed on 20)
    - Requires: `sudo apt install cmake-curses-gui`
	- Build:
	```
	mkdir build
	cd build
	cmake ..
	make
	```
- CentOS 7:
    - Requires: `sudo yum install cmake3`
	- Build:
	```
	mkdir build
	cd build
	cmake3 ..
	make
	```
	(mind to use `cmake3`, not `cmake`, using the wrong one will mess up the build directory)

  In case of errors:

    - Make sure that the cuda binary and library are included in $PATH and $LD_LIBRARY_PATH:
    ```
    export PATH="/usr/local/cuda-10.1/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH"
    ```

  - Set the OptiX_INSTALL_DIR to point to the installation of the Optix SDK
   ```
    export OptiX_INSTALL_DIR=/..../NVIDIA-OptiX-SDK-7.2.0-linux64-x86_64
    ```

  - Modify Line 54 of [CMakeLists.txt](https://github.com/vani-nag/OWLRayTracing/blob/master/CMakeLists.txt) to 
  `set (CMAKE_CXX_FLAGS "--std=c++11 -pthread")`. Note the inclusion of `-pthread`.
- Windows
    - Requires: Visual Studio (both 2017 and 2019 work), OptiX 7.0, cmake
	- Build: Use CMake-GUI to build Visual Studio project, then use VS to build
		- Specifics: source code path is ```...Gitlab/owl```, binaries ```...Gitlab/owl/build```, and after pushing the Configure button choose ```x64``` for the optional platform.
		- You may need to Configure twice.
		- If you get "OptiX headers (optix.h and friends) not found." then define OptiX_INCLUDE manually in CMake-gui by setting it to ```C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0/include```

<!--- ------------------------------------------------------- -->
# Using OWL through CMake

Though you can of course use OWL without CMake, it is highly encouraged
to use OWL as a git submodule, using CMake to configure and build this
submodule. In particular, the suggested procedure is to first
do a `add_subdirectory` with the owl submodules as such:

    set(owl_dir ${PROJECT_SOURCE_DIR}/whereeverYourOWLSubmoduleIs)
    add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

(the `EXCLUDE_FROM_ALL` makes sure that your main project won't
automatically build any owl samples or test cases unless you explicitly request so).

After that `include_subdirectory` OWL sets some CMake variables in the
parent script that let this parent CMakeList file use it as if it had been
found with a `find_package` script. In particular, it will set the following variables for the user's convenience:

- `OWL_INCLUDES`: the list of directories where owl-related includes
  (like `owl/owl.h' etc) can be found.

- `OWL_CXX_FLAGS`: command-line parameters that the app should pass as
  `compile_definitions` to any source file that includes any owl
  header files. 

- `OWL_LIBRARIES`: the list of libraries the user should link to when
  using OWL.  This will, for example, automatically include TBB
  dependencies if those could be found.
  
- `OWL_VIEWER_LIBRARIES`: libraries required when (also) using the OWL
  sampler viewer widget (programs that use their own viewer/windowing
  code can ignore this).
  
For sample code on how to use this, have a look at the `owl/samples/`
directory.

<!--- ------------------------------------------------------- -->
#### Copyright
&copy; Purdue Research Foundation

#### Non-commercial Use License
[Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)