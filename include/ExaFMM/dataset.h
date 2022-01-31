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
#ifndef INCLUDE_EXAFMM_DATASET_H_
#define INCLUDE_EXAFMM_DATASET_H_
#include <random>

#include "exafmm.h"

namespace ExaFMM {
	/**
	 * @brief Generate uniform distribution in a cube from 0 to 1.
	 *
	 * @tparam T Target's value type (real or complex).
	 * @param numBodies Number of bodies.
	 * @param seed Seed of pseudorandom number generator.
	 * @return Bodies Vector of bodies.
	 */
	template <typename T>
	Bodies<T> cube(int numBodies, int /* seed */) {
		Bodies<T> bodies(numBodies);
		std::uniform_real_distribution<double> uniformDist(0., 1.);
		std::default_random_engine randomEngine;
		for (int b = 0; b < numBodies; b++) {
			for (int d = 0; d < 3; d++) {
				bodies[b].X[d] = uniformDist(randomEngine);
			}
		}
		return bodies;
	}

	/**
	 * @brief Generate uniform distribution in a sphere with a radius of 1.
	 *
	 * @tparam T Target's value type (real or complex).
	 * @param numBodies Number of bodies.
	 * @param seed Seed of pseudorandom number generator.
	 * @return Bodies Vector of bodies.
	 */
	template <typename T>
	Bodies<T> sphere(int numBodies, int /*seed*/) {
		Bodies<T> bodies(numBodies);
		std::uniform_real_distribution<double> uniformDist(0., 1.);
		std::default_random_engine randomEngine;
		for (int b = 0; b < numBodies; b++) {
			for (int d = 0; d < 3; d++) {
				bodies[b].X[d] = uniformDist(randomEngine) * 2 - 1;
			}
			real_t r = std::sqrt(bodies[b].X.norm());
			bodies[b].X /= r;
		}
		return bodies;
	}

	/**
	 * @brief Generate plummer distribution in a cube from 0 to 1.
	 *
	 * @tparam T Target's value type (real or complex).
	 * @param numBodies Number of bodies.
	 * @param seed Seed of pseudorandom number generator.
	 * @return Bodies Vector of bodies.
	 */
	template <typename T>
	Bodies<T> plummer(int numBodies, int seed) {
		Bodies<T> bodies(numBodies);
		std::uniform_real_distribution<double> uniformDist(0., 1.);
		std::default_random_engine randomEngine;
		int i = 0;
		int Xmax = 0;
		while (i < numBodies) {
			real_t X1 = uniformDist(randomEngine);
			real_t X2 = uniformDist(randomEngine);
			real_t X3 = uniformDist(randomEngine);
			real_t R = 1.0 / sqrt((pow(X1, -2.0 / 3.0) - 1.0));
			if (R < 100) {
				real_t Z = (1.0 - 2.0 * X2) * R;
				real_t X = sqrt(R * R - Z * Z) * std::cos(2.0 * PI * X3);
				real_t Y = sqrt(R * R - Z * Z) * std::sin(2.0 * PI * X3);
				bodies[i].X[0] = X;
				bodies[i].X[1] = Y;
				bodies[i].X[2] = Z;
				for (int d = 0; d < 3; d++) {
					Xmax = Xmax > fabs(bodies[i].X[d]) ? Xmax : static_cast<int>(fabs(bodies[i].X[d]));
				}
				i++;
			}
		}
		real_t scale = 0.5 / (Xmax + 1);
		for (i = 0; i < numBodies; i++) {
			for (int d = 0; d < 3; d++) {
				bodies[i].X[d] = bodies[i].X[d] * scale + 0.5;
			}
		}
		return bodies;
	}

	/**
	 * @brief Generate targets with various distributions.
	 *
	 * @tparam T Target's value type (real or complex).
	 * @param numBodies Number of bodies.
	 * @param distribution Type of distribution: 'c' for cube, 's' for sphere, 'p'
	 * for plummer.
	 * @param seed Seed of pseudorandom number generator.
	 * @return Bodies Vector of targets.
	 */
	template <typename T>
	Bodies<T> init_targets(int numBodies, const char* distribution, int seed) {
		Bodies<T> bodies;
		switch (distribution[0]) {
		case 'c':
			bodies = cube<T>(numBodies, seed);
			break;
		case 'p':
			bodies = plummer<T>(numBodies, seed);
			break;
		case 's':
			bodies = sphere<T>(numBodies, seed);
			break;
		default:
			fprintf(stderr, "Unknown data distribution %s\n", distribution);
		}
		for (int i = 0; i < numBodies; ++i) {
			bodies[i].ibody = i;
		}
		return bodies;
	}

	/**
	 * @brief Generate sources with various distributions.
	 *
	 * @tparam T Source's value type (real or complex).
	 * @param numBodies Number of bodies.
	 * @param distribution Type of distribution: 'c' for cube, 's' for sphere, 'p'
	 * for plummer.
	 * @param seed Seed of pseudorandom number generator.
	 * @return Bodies Vector of sources.
	 */
	template <typename T>
	Bodies<T> init_sources(int numBodies, const char* distribution, int seed) {
		Bodies<T> bodies = init_targets<T>(numBodies, distribution, seed);
		std::uniform_real_distribution<double> uniformDist(0., 1.);
		std::default_random_engine randomEngine;
		for (int b = 0; b < numBodies; ++b) {
			bodies[b].q = uniformDist(randomEngine) - 0.5;
		}
		return bodies;
	}

	// Template specialization of init_source for complex type
	template <>
	Bodies<complex_t> init_sources(int numBodies, const char* distribution,
		int seed) {
		std::uniform_real_distribution<double> uniformDist(0., 1.);
		std::default_random_engine randomEngine;
		Bodies<complex_t> bodies =
			init_targets<complex_t>(numBodies, distribution, seed);
		for (int b = 0; b < numBodies; ++b) {
			bodies[b].q = complex_t(uniformDist(randomEngine) - 0.5,
				uniformDist(randomEngine) - 0.5);
		}
		return bodies;
	}
}  // namespace ExaFMM
#endif  // INCLUDE_EXAFMM_DATASET_H_
