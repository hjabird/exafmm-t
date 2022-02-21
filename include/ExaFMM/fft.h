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
#include <complex>

namespace ExaFMM {

enum class fft_dir{
	forwards,
	backwards,
};

template<typename PotentialT, fft_dir Direction>
class fft {
protected:
	static constexpr bool isComplex = is_complex<PotentialT>::value;
	static constexpr bool isFloat = isComplex ? 
		std::is_same_v<PotentialT, std::complex<float>> :
		std::is_same_v<PotentialT, float>;

	using fftw_complex_t = std::conditional_t<
		isFloat, fftwf_complex, fftw_complex>;
public:
	static constexpr fft_dir direction{Direction};

	using complex_t = std::conditional_t<isFloat, 
		std::complex<float>, std::complex<double>>;

	using real_t = std::conditional_t<isFloat, float, double>;

	using plan_t = std::conditional_t<isFloat, fftwf_plan, fftw_plan>;

	using in_t = std::conditional_t<
		Direction == fft_dir::backwards || isComplex,
		complex_t, real_t>;

	using out_t = std::conditional_t<
		Direction == fft_dir::forwards || isComplex,
		complex_t, real_t>;

	static plan_t plan_many(int rank,  const int* n,int howMany,
		in_t* in, const int* inEmbed, int inStride, int inDist, 
		out_t* out, const int* outEmbed, int outStride, int outDist) {
		if constexpr (isFloat) {
			if constexpr (Direction == fft_dir::forwards) {
				if constexpr (isComplex) {
					return fftwf_plan_many_dft(rank,  n, howMany,
						reinterpret_cast<fftw_complex_t*>(in), 
						inEmbed, inStride, inDist,
						reinterpret_cast<fftw_complex_t*>(out), 
						outEmbed, outStride, outDist,
						FFTW_FORWARD, FFTW_ESTIMATE);
				}
				else { // !isComplex
					return fftwf_plan_many_dft_r2c(rank, n, howMany,
						in, inEmbed, inStride, inDist,
						reinterpret_cast<fftw_complex_t*>(out),
						outEmbed, outStride, outDist,
						FFTW_ESTIMATE);
				}
			}
			else { // fft_dir == fft_dir::backwards
				if constexpr (isComplex) {
					return fftwf_plan_many_dft(rank, n, howMany,
						reinterpret_cast<fftw_complex_t*>(in),
						inEmbed, inStride, inDist,
						reinterpret_cast<fftw_complex_t*>(out),
						outEmbed, outStride, outDist,
						FFTW_BACKWARD, FFTW_ESTIMATE);
				}
				else { // !isComplex
					return fftwf_plan_many_dft_c2r(rank, n, howMany,
						reinterpret_cast<fftw_complex_t*>(in),
						inEmbed, inStride, inDist,
						out, outEmbed, outStride, outDist,
						FFTW_ESTIMATE);
				}
			}
		}
		else { // Double
			if constexpr (Direction == fft_dir::forwards) {
				if constexpr (isComplex) {
					return fftw_plan_many_dft(rank, n, howMany,
						reinterpret_cast<fftw_complex_t*>(in),
						inEmbed, inStride, inDist,
						reinterpret_cast<fftw_complex_t*>(out),
						outEmbed, outStride, outDist,
						FFTW_FORWARD, FFTW_ESTIMATE);
				}
				else { // !isComplex
					return fftw_plan_many_dft_r2c(rank, n, howMany,
						in, inEmbed, inStride, inDist,
						reinterpret_cast<fftw_complex_t*>(out),
						outEmbed, outStride, outDist,
						FFTW_ESTIMATE);
				}
			}
			else { // fft_dir == fft_dir::backwards
				if constexpr (isComplex) {
					return fftw_plan_many_dft(rank, n, howMany,
						reinterpret_cast<fftw_complex_t*>(in),
						inEmbed, inStride, inDist,
						reinterpret_cast<fftw_complex_t*>(out),
						outEmbed, outStride, outDist,
						FFTW_BACKWARD, FFTW_ESTIMATE);
				}
				else { // !isComplex
					return fftw_plan_many_dft_c2r(rank, n, howMany,
						reinterpret_cast<fftw_complex_t*>(in), 
						inEmbed, inStride, inDist,
						out, outEmbed, outStride, outDist,
						FFTW_ESTIMATE);
				}
			}
		}
	}

	static plan_t plan_one(int rank, const int* n, in_t* in, out_t* out) {
		if constexpr (isFloat) {
			if constexpr (Direction == fft_dir::forwards) {
				if constexpr (isComplex) {
					return fftwf_plan_dft(rank, n,
						reinterpret_cast<fftw_complex_t*>(in),
						reinterpret_cast<fftw_complex_t*>(out), 
						FFTW_FORWARD, FFTW_ESTIMATE);
				}
				else { // !isComplex
					return fftwf_plan_dft_r2c(rank, n, in,
						reinterpret_cast<fftw_complex_t*>(out), FFTW_ESTIMATE);
				}
			}
			else { // fft_dir == fft_dir::backwards
				if constexpr (isComplex) {
					return fftwf_plan_dft(rank, n,
						reinterpret_cast<fftw_complex_t*>(in),
						reinterpret_cast<fftw_complex_t*>(out), ,
						FFTW_BACKWARD, FFTW_ESTIMATE);
				}
				else { // !isComplex
					return fftwf_plan_dft_c2r(rank, n, in,
						reinterpret_cast<fftw_complex_t*>(out), FFTW_ESTIMATE);
				}
			}
		}
		else { // Double
			if constexpr (Direction == fft_dir::forwards) {
				if constexpr (isComplex) {
					return fftw_plan_dft(rank, n,
						reinterpret_cast<fftw_complex_t*>(in),
						reinterpret_cast<fftw_complex_t*>(out),
						FFTW_FORWARD, FFTW_ESTIMATE);
				}
				else { // !isComplex
					return fftw_plan_dft_r2c(rank, n, in,
						reinterpret_cast<fftw_complex_t*>(out), FFTW_ESTIMATE);
				}
			}
			else { // fft_dir == fft_dir::backwards
				if constexpr (isComplex) {
					return fftw_plan_dft(rank, n,
						reinterpret_cast<fftw_complex_t*>(in),
						reinterpret_cast<fftw_complex_t*>(out),
						FFTW_BACKWARD, FFTW_ESTIMATE);
				}
				else { // !isComplex
					return fftw_plan_dft_c2r(rank, n, in,
						reinterpret_cast<fftw_complex_t*>(out), FFTW_ESTIMATE);
				}
			}
		}
	}

	static void execute(plan_t plan, in_t* in, out_t* out) {
		if constexpr (isComplex) {
			if constexpr (isFloat) {
				fftwf_execute_dft(plan,
					reinterpret_cast<fftw_complex_t*>(in),
					reinterpret_cast<fftw_complex_t*>(out));
			}
			else {
				fftw_execute_dft(plan,
					reinterpret_cast<fftw_complex_t*>(in),
					reinterpret_cast<fftw_complex_t*>(out));
			}
		}
		else if constexpr (isFloat) { // !isComplex
			if constexpr (Direction == fft_dir::forwards) {
				fftwf_execute_dft_r2c(plan, in,
					reinterpret_cast<fftw_complex_t*>(out));
			}
			else { // fft_dir == fft_dir::backwards
				fftwf_execute_dft_c2r(plan,
					reinterpret_cast<fftw_complex_t*>(in), out);
			}
		}
		else { // Double
			if constexpr (Direction == fft_dir::forwards) {
				fftw_execute_dft_r2c(plan, in,
					reinterpret_cast<fftw_complex_t*>(out));
			}
			else { // fft_dir == fft_dir::backwards
				fftw_execute_dft_c2r(plan,
					reinterpret_cast<fftw_complex_t*>(in), out);
			}
		}
	}

	static void destroy_plan(plan_t plan) {
		if constexpr (isFloat) {
			fftwf_destroy_plan(plan);
		}
		else {
			fftw_destroy_plan(plan);
		}
	}
};
}  // namespace ExaFMM

#endif  // INCLUDE_EXAFMM_FFT_H_