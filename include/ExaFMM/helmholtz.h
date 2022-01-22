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
#ifndef INCLUDE_EXAFMM_HELMHOLTZ_H_
#define INCLUDE_EXAFMM_HELMHOLTZ_H_
#include "exafmm.h"
#include "fmm.h"
#include "geometry.h"
#include "intrinsics.h"
#include "timer.h"

namespace ExaFMM {

class HelmholtzFmmKernel;

using HelmholtzFmm = Fmm<HelmholtzFmmKernel>;

//! A class defining the kernel function for the Helmholtz FMM.
class HelmholtzFmmKernel {
 public:
  using potential_t = std::complex<double>;

 private:
  using pt = potential_traits<potential_t>;
  template <int Rows = dynamic>
  using potential_vector_t = typename pt::template potential_vector_t<Rows>;
  using complex_t = typename pt::complex_t;
  using coord_t = typename pt::coord_t;
  template <int Rows = dynamic, int RowOrder = column_major>
  using coord_matrix_t = typename pt::template coord_matrix_t<Rows, RowOrder>;

 public:
  // Argument types required for this kernel.
  using kernel_args_t = std::tuple<complex_t>;

  const complex_t wavek;  //!< Wave number k.

 private:
  const complex_t kappa;  //!< K used in Helmholtz kernel.
  const real_t kernelCoef;

 public:
  HelmholtzFmmKernel() = delete;

  HelmholtzFmmKernel(kernel_args_t kernelArgs)
      : wavek{std::get<0>(kernelArgs)},
        kappa{wavek / complex_t{16}},
        kernelCoef{real_t(1) / (64 * PI)} {};

  /** Compute the effect of source with a given strength at a given coordinate.
   * @param sourceCoord The coordinate of the source particle.
   * @param targetCoord The measurement location.
   * @return The potential measured at targetCoord.
   **/
  inline potential_t potential_P2P(const coord_t sourceCoord,
                                   const coord_t targetCoord) const noexcept {
    auto radius = (sourceCoord - targetCoord).norm();
    return kernelCoef * std::exp(radius * kappa * complex_t{0, 1}) / radius;
  }

  /** Compute the effect of source with a given strength at a given coordinate.
   * @param sourceCoord The coordinate of the source particle.
   * @param sourceStrength The strength of the source particle.
   * @param targetCoord The measurement location.
   * @return The potential measured at targetCoord.
   **/
  inline potential_t potential_P2P(const coord_t sourceCoord,
                                   const potential_t sourceStrength,
                                   const coord_t targetCoord) const noexcept {
    return sourceStrength * potential_P2P(sourceCoord, targetCoord);
  }

  template <int NumSources, int NumTargets, int SourceRowOrder,
            int TargetRowOrder>
  potential_vector_t<NumTargets> potential_P2P(
      const coord_matrix_t<NumSources, SourceRowOrder>& sourceCoords,
      const potential_vector_t<NumSources>& sourceStrengths,
      const coord_matrix_t<NumTargets, TargetRowOrder>& targetCoords) {
    simdvec zero((real_t)0);
    const int numSources = sourceCoords.rows();
    const int numTargets = targetCoords.rows();
    auto targetValues = potential_vector_t<NumTargets>::Zero(numTargets);
    for (size_t i{0}; i < numTargets; ++i) {
      for (size_t j{0}; j < numSources; ++j) {
        targetValues(i) += potential_P2P(
            sourceCoords.row(j), sourceStrengths(j), targetCoords.row(i));
      }
    }
    return targetValues;
  }

  /** Compute potentials and gradients at targets induced by sources
   * directly.
   * @param sourceCoords Vector of coordinates of sources.
   * @param targetCoords Vector of coordinates of targets.
   * @return Vector of potentials of targets.
   */
  inline potential_t gradient_P2P(const coord_t sourceCoord,
                                  const coord_t targetCoord) const noexcept {
    auto radius = (sourceCoord - targetCoord).norm();
    auto newtonOffset{1 / (16 * 16)};
  }

  /** Compute potentials and gradients at targets induced by sources
   * directly.
   *
   * @param sourceCoords Vector of coordinates of sources.
   * @param sourceStrengths Vector of charges of sources.
   * @param targetCoords Vector of coordinates of targets.
   * @return Vector of potentials of targets.
   */
  template <int NumSources, int NumTargets, int SourceRowOrder,
            int TargetRowOrder>
  potential_vector_t<dynamic> gradient_P2P(
      const coord_matrix_t<NumSources, SourceRowOrder>& sourceCoords,
      const potential_vector_t<NumSources>& sourceStrengths,
      const coord_matrix_t<NumTargets, TargetRowOrder>& targetCoords) {
    // The output is a mixture of potential and gradient... What to do?
    simdvec zero((real_t)0);
    simdvec one((real_t)1);
    real_t newton_coef =
        16;  // comes from Newton's method in simd rsqrt function
    simdvec coef(real_t(1.0 / (4 * PI * newton_coef)));
    simdvec k_real(wavek.real() / newton_coef);
    simdvec k_imag(wavek.imag() / newton_coef);
    simdvec newton_offset(
        real_t(1.0 / (newton_coef * newton_coef)));  // offset invr2 term
    int nsrcs = src_coord.size() / 3;
    int ntrgs = trg_coord.size() / 3;
    int t;
    const complex_t I(0, 1);
    for (t = 0; t + NSIMD <= ntrgs; t += NSIMD) {
      simdvec tx(&trg_coord[3 * t + 0], 3 * (int)sizeof(real_t));
      simdvec ty(&trg_coord[3 * t + 1], 3 * (int)sizeof(real_t));
      simdvec tz(&trg_coord[3 * t + 2], 3 * (int)sizeof(real_t));
      simdvec tv0_real(zero);
      simdvec tv0_imag(zero);
      simdvec tv1_real(zero);
      simdvec tv1_imag(zero);
      simdvec tv2_real(zero);
      simdvec tv2_imag(zero);
      simdvec tv3_real(zero);
      simdvec tv3_imag(zero);
      for (int s = 0; s < nsrcs; s++) {
        simdvec sx(src_coord[3 * s + 0]);
        sx -= tx;  // sx is negative dx
        simdvec sy(src_coord[3 * s + 1]);
        sy -= ty;
        simdvec sz(src_coord[3 * s + 2]);
        sz -= tz;
        simdvec sv_real(src_value(s).real());
        simdvec sv_imag(src_value(s).imag());
        simdvec r2(zero);
        r2 += sx * sx;
        r2 += sy * sy;
        r2 += sz * sz;
        simdvec invr = rsqrt(r2);
        invr &= r2 > zero;

        simdvec r = r2 * invr;
        simdvec kr_real = k_real * r;   // k_real * r
        simdvec kr_imag = k_imag * r;   // k_imag * r
        simdvec e_ikr = exp(-kr_imag);  // exp(-k_imag*r)
        // simdvec kr = k * r2 * invr;   // newton_coefs in k & invr cancel out
        simdvec G_real = e_ikr * cos(kr_real) * invr;  // G = e^(ikr) / r
        simdvec G_imag =
            e_ikr * sin(kr_real) * invr;  // invr carries newton_coef
        simdvec potential_real =
            sv_real * G_real - sv_imag * G_imag;  // p = G * q
        simdvec potential_imag = sv_real * G_imag + sv_imag * G_real;
        tv0_real += potential_real;
        tv0_imag += potential_imag;

        // coefg = (1+k_imag*r)/r2 - k_real/r*I (considering the negative sign
        // in sx)
        simdvec coefg_real = (one + kr_imag) * invr * invr * newton_offset;
        simdvec coefg_imag = -k_real * invr;
        // gradient = coefg * potential * dx
        simdvec gradient_real =
            coefg_real * potential_real - coefg_imag * potential_imag;
        simdvec gradient_imag =
            coefg_real * potential_imag + coefg_imag * potential_real;
        tv1_real += sx * gradient_real;
        tv1_imag += sx * gradient_imag;
        tv2_real += sy * gradient_real;
        tv2_imag += sy * gradient_imag;
        tv3_real += sz * gradient_real;
        tv3_imag += sz * gradient_imag;
      }
      tv0_real *= coef;
      tv0_imag *= coef;
      tv1_real *= coef;
      tv1_imag *= coef;
      tv2_real *= coef;
      tv2_imag *= coef;
      tv3_real *= coef;
      tv3_imag *= coef;
      for (int m = 0; m < NSIMD && (t + m) < ntrgs; m++) {
        trg_value(4 * (t + m) + 0) += complex_t(tv0_real[m], tv0_imag[m]);
        trg_value(4 * (t + m) + 1) += complex_t(tv1_real[m], tv1_imag[m]);
        trg_value(4 * (t + m) + 2) += complex_t(tv2_real[m], tv2_imag[m]);
        trg_value(4 * (t + m) + 3) += complex_t(tv3_real[m], tv3_imag[m]);
      }
    }
    for (; t < ntrgs; t++) {
      complex_t potential(0, 0);
      cvec3 gradient = complex_t(0., 0.);
      for (int s = 0; s < nsrcs; s++) {
        vec3 dx;
        for (int d = 0; d < 3; d++)
          dx[d] = trg_coord[3 * t + d] - src_coord[3 * s + d];
        real_t r2 = norm(dx);
        if (r2 != 0) {
          real_t r = std::sqrt(r2);
          complex_t potential_ij = std::exp(I * r * wavek) * src_value(s) / r;
          complex_t coefg = (I * wavek / r - 1 / r2) * potential_ij;
          potential += potential_ij;
          for (int d = 0; d < 3; d++) gradient[d] += coefg * dx[d];
        }
      }
      trg_value(4 * t + 0) += potential / (4 * PI);
      trg_value(4 * t + 1) += gradient[0] / (4 * PI);
      trg_value(4 * t + 2) += gradient[1] / (4 * PI);
      trg_value(4 * t + 3) += gradient[2] / (4 * PI);
    }
  }
};
}  // namespace ExaFMM
#endif  // INCLUDE_EXAFMM_HELMHOLTZ_H_
