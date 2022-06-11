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
#ifndef INCLUDE_MFMM_PREDEFINES_H_
#define INCLUDE_MFMM_PREDEFINES_H_

#include <cassert>

/** MFMM_ASSERT(X) allows for simple assertions in debug mode or compiler
 * assumptions in release mode.
 **/
#ifdef _MSC_VER

#ifdef _DEBUG
#define MFMM_ASSERT(X) assert(X)
#else
#define MFMM_ASSERT(X) __assume(X)
#endif

#endif

#endif  // INCLUDE_MFMM_PREDEFINES_H_
