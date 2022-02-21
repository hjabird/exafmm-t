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

template<typename PotentialT>
class fft {
protected:
	static constexpr bool isComplex = is_complex<PotentialT>::value;
	static constexpr bool isFloat = isComplex ? 
		std::is_same_v<PotentialT, std::complex<float>> :
		std::is_same_v<PotentialT, float>;

	using fftw_complex_t = std::conditional_t<
		isFloat, fftwf_complex, fftw_complex>;
public:
	using complex_t = std::conditional_t<isFloat, 
		std::complex<float>, std::complex<double>>;

	using real_t = std::conditional_t<isFloat, float, double>;

	using plan_t = std::conditional_t<isFloat, fftwf_plan, fftw_plan>;

	static plan_t plan_dft(int rank, const int* n, complex_t* in,complex_t* out,
			int sign, unsigned int flags) {
		if constexpr (fft<PotentialT>::isFloat) {
			return fftwf_plan_dft(rank, n, 
				reinterpret_cast<fftw_complex_t*>(in), 
				reinterpret_cast<fftw_complex_t*>(out), sign, flags);
		}
		else { // Double
			return fftw_plan_dft(rank, n, 
				reinterpret_cast<fftw_complex_t*>(in), 
				reinterpret_cast<fftw_complex_t*>(out), sign, flags);
		}
	}

	static plan_t plan_many_dfts(int rank,  const int* n,int howMany,
		complex_t* in, const int* inEmbed, int inStride, int inDist, 
		complex_t* out, const int* outEmbed, int outStride, int outDist,
		int sign, unsigned int flags) {
		if constexpr (fft<PotentialT>::isFloat) {
			return fftwf_plan_many_dft(rank,  n, howMany,
				reinterpret_cast<fftw_complex_t*>(in), 
				inEmbed, inStride, inDist,
				reinterpret_cast<fftw_complex_t*>(out), 
				outEmbed, outStride, outDist,
				sign, flags);
		}
		else { // Double
			return fftw_plan_many_dft(rank, n, howMany,
				reinterpret_cast<fftw_complex_t*>(in), 
				inEmbed, inStride, inDist,
				reinterpret_cast<fftw_complex_t*>(out), 
				outEmbed, outStride, outDist,
				sign, flags);
		}
	}

	static void execute_dft(plan_t plan, complex_t* in, complex_t* out) {
		if constexpr (fft<PotentialT>::isFloat) {
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

	static plan_t plan_dft_r2c(int rank, const int* n, real_t* in, complex_t* out,
		unsigned int flags) {
		if constexpr (fft<PotentialT>::isFloat) {
			return fftwf_plan_dft_r2c(rank, n, in, 
				reinterpret_cast<fftw_complex_t*>(out), flags);
		}
		else {
			return fftw_plan_dft_r2c(rank, n, in, 
				reinterpret_cast<fftw_complex_t*>(out), flags);
		}
	}

	static plan_t plan_many_dfts_r2c(int rank, const int* n, int howMany,
		real_t* in, const int* inEmbed, int inStride, int inDist,
		complex_t* out, const int* outEmbed, int outStride, int outDist,
		unsigned int flags) {
		if constexpr (fft<PotentialT>::isFloat) {
			fftw_plan_many_dft_c2r()
			return fftwf_plan_many_dft_r2c(rank, n, howMany,
				in, inEmbed, inStride, inDist,
				reinterpret_cast<fftw_complex_t*>(out), 
				outEmbed, outStride, outDist,
				flags);
		}
		else {
			return fftw_plan_many_dft_r2c(rank, n, howMany,
				in, inEmbed, inStride, inDist,
				reinterpret_cast<fftw_complex_t*>(out), 
				outEmbed, outStride, outDist,
				flags);
		}
	}

	static void execute_dft_r2c(plan_t plan, real_t* in, complex_t* out) {
		if constexpr (fft<PotentialT>::isFloat) {
			fftwf_execute_dft_r2c(plan, in, 
				reinterpret_cast<fftwf_complex_t*>(out));
		}
		else {
			fftw_execute_dft_r2c(plan, in, 
				reinterpret_cast<fftw_complex_t*>(out));
		}
	}

	static plan_t plan_dft_c2r(int rank, const int* n, real_t* in, complex_t* out,
		unsigned int flags) {
		if constexpr (fft<PotentialT>::isFloat) {
			return fftwf_plan_dft_c2r(rank, n, in, out, flags);
		}
		else {
			return fftw_plan_dft_c2r(rank, n, in, out, flags);
		}
	}

	static plan_t plan_many_dfts_c2r(int rank, const int* n, int howMany,
		complex_t* in, const int* inEmbed, int inStride, int inDist,
		real_t* out, const int* outEmbed, int outStride, int outDist,
		unsigned int flags) {
		if constexpr (fft<PotentialT>::isFloat) {
			return fftwf_plan_many_dft_c2r(rank, n, howMany,
				reinterpret_cast<fftw_complex_t*>(in), inEmbed, inStride, inDist,
				out, outEmbed, outStride, outDist,
				flags);
		}
		else {
			return fftw_plan_many_dft_c2r(rank, n, howMany,
				reinterpret_cast<fftw_complex_t*>(in), inEmbed, inStride, inDist,
				out, outEmbed, outStride, outDist,
				flags);
		}
	}

	static void execute_dft_c2r(plan_t plan, complex_t* in, real_t* out) {
		if constexpr (fft<PotentialT>::isFloat) {
			fftwf_execute_dft_c2r(plan, 
				reinterpret_cast<fftw_complex_t*>(in), out);
		}
		else {
			fftw_execute_dft_c2r(plan, 
				reinterpret_cast<fftw_complex_t*>(in), out);
		}
	}

	static void destroy_plan(plan_t plan) {
		if constexpr (fft<PotentialT>::isFloat) {
			fftwf_destroy_plan(plan);
		}
		else {
			fftw_destroy_plan(plan);
		}
	}
};
}  // namespace ExaFMM

#endif  // INCLUDE_EXAFMM_FFT_H_