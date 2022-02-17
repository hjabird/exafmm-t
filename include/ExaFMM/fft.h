#pragma once
/******************************************************************************
 *
 * ExaFMM
 * A high-performance fast multipole method library with C++ and
 * python interfaces.
 *
 * Originally Tingyu Wang, Rio Yokota and Lorena A. Barba
 * Modified by HJA Bird
 *
 ******************************************************************************/
#ifndef INCLUDE_EXAFMM_FFT_H_
#define INCLUDE_EXAFMM_FFT_H_

#include <fftw3.h>

namespace ExaFMM {
#if FLOAT
typedef float real_t;  //!< Real number type
const real_t EPS = 1e-8f;
typedef fftwf_complex fft_complex;
typedef fftwf_plan fft_plan;
#define fft_plan_dft fftwf_plan_dft
#define fft_plan_many_dft fftwf_plan_many_dft
#define fft_execute_dft fftwf_execute_dft
#define fft_plan_dft_r2c fftwf_plan_dft_r2c
#define fft_plan_many_dft_r2c fftwf_plan_many_dft_r2c
#define fft_plan_many_dft_c2r fftwf_plan_many_dft_c2r
#define fft_execute_dft_r2c fftwf_execute_dft_r2c
#define fft_execute_dft_c2r fftwf_execute_dft_c2r
#define fft_destroy_plan fftwf_destroy_plan
#define fft_flops fftwf_flops
#else
using real_t = double;  //!< Real number type
const real_t EPS = 1e-16;
using fft_complex = fftw_complex;
using fft_plan = fftw_plan;
#define fft_plan_dft fftw_plan_dft
#define fft_plan_many_dft fftw_plan_many_dft
#define fft_execute_dft fftw_execute_dft
#define fft_plan_dft_r2c fftw_plan_dft_r2c
#define fft_plan_many_dft_r2c fftw_plan_many_dft_r2c
#define fft_plan_many_dft_c2r fftw_plan_many_dft_c2r
#define fft_execute_dft_r2c fftw_execute_dft_r2c
#define fft_execute_dft_c2r fftw_execute_dft_c2r
#define fft_destroy_plan fftw_destroy_plan
#define fft_flops fftw_flops
#endif
} // namespace ExaFMM

#endif // INCLUDE_EXAFMM_FFT_H_