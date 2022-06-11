#pragma once
/******************************************************************************
 *
 * mfmm
 * A high-performance fast multipole method library using C++.
 *
 * A fork of ExaFMM (BSD-3-Clause lisence).
 * Originally copyright Wang, Yokota and Barba.
 *
 * Modifications copyright HJA Bird.
 *
 ******************************************************************************/
#ifndef INCLUDE_MFMM_FFT_H_
#define INCLUDE_MFMM_FFT_H_

#include <fftw3.h>
#include <complex>

namespace mfmm {

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

	/** Construct a plan for many FFTs. Plan stored internally to fft object.
	* @param rank The dimensionality of the FFT.
	* @param n Size of transform to compute.
	* @param howMany How many FFTs will be executed?
	* @param inDist The input size per FFT.
	* @param outDist The output size per FFT.
	**/
	fft(int rank, const int* n, int howMany,
		int inDist, int outDist) {
		std::vector<in_t> inData(inDist * howMany);
		std::vector<out_t> outData(outDist * howMany);
		m_plan = plan_many(rank, n, howMany, inData.data(), nullptr, 1, inDist,
			outData.data(), nullptr, 1, outDist);
	}

	/** Construct a plan for one FFTs. Plan stored internally to fft object.
	* @param rank The dimensionality of the FFT.
	* @param n Size of transform to compute.
	**/
	fft(int rank, const int* n) : fft(rank, n, 1, 1, 1) {};

	~fft() {
		if constexpr (isFloat) {
			fftwf_destroy_plan(m_plan);
		}
		else {
			fftw_destroy_plan(m_plan);
		}
	}

	void execute(in_t* in, out_t* out) {
		auto& plan = m_plan;
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

protected:
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

	protected:
		plan_t m_plan;
};
}  // namespace mfmm

#endif  // INCLUDE_MFMM_FFT_H_