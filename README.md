# A fork of ExaFMM-T

This is fork of the ExaFMM-t project and very much a work in progress. So what are the big changes so far? The good:
* Improved comments and code clarity.
* Easier to compile everywhere. Developed with MSVC, but intended to compile everywhere at some point.
* Eigen is now used for linear algebra.
  * No longer dependent on BLAS or LAPACK libaries.
  * Everything is header-only.
  * This makes the code easier to read.
* No more virtual. Theoretically improved performance.
* All AMD64 intrinsics have been removed.
  * Should compile on non-AMD64 architectures.
* Heavier use of C++17:
  * All types are now handled using templates - no macros required.
  * Templated sections of the code no longer need type specific specializations.
  * Constexpr allows more to be done at compile time:
    * Some array sizes are now known at compile time.
	* Some pre-computation can occur at compile time.

The bad:
* The python interface is gone.
* This code still needs to be validated.

To do:
* Change interface for ease of use.
* Improved testing.
* Template based on FMM order.
* Octree representation
  * Source leaves, target leaves, source nodes and target nodes should all be different types.
* Regularised particles.
* Vector potentials.
* GPU acceleration:
  * SYCL-2020 (for this, the codebase must stay as C++2017).
* Non-particle sources.

## What was exafmm-t

The original exafmm-t is available [here](https://github.com/exafmm/exafmm-t) and no longer appears to be actively developed.

[![status](https://joss.theoj.org/papers/0faabca7e0ef645b42d7dd72cc924ecc/status.svg)](https://joss.theoj.org/papers/0faabca7e0ef645b42d7dd72cc924ecc)

"exafmm: a high-performance fast multipole method library with C++ and Python interfaces", Tingyu Wang, Rio Yokota, Lorena A. Barba. The Journal of Open Source Software, 6(61):3145 (2021). doi:10.21105/joss.03145 

**exafmm-t** is a kernel-independent fast multipole method library for solving N-body problems.
It provides both C++ and Python APIs.
We use [pybind11](https://github.com/pybind/pybind11) to create Python bindings from C++ code.
exafmm-t aims to deliver compelling performance with a simple code design and a user-friendly interface.
It currently supports both potential and force calculation of Laplace, low-frequency Helmholtz and modified Helmholtz (Yukawa) kernel in 3D.
In addition, users can easily add other non-oscillatory kernels under exafmm-t's framework.

The full documentation is available [here](https://exafmm.github.io/exafmm-t).
