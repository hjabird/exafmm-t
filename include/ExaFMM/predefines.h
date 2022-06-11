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
#ifndef INCLUDE_EXAFMM_PREDEFINES_H_
#define INCLUDE_EXAFMM_PREDEFINES_H_

#include <cassert>

/** EXAFMM_ASSERT(X) allows for simple assertions in debug mode or compiler
 * assumptions in release mode.
 **/
#ifdef _MSC_VER

#ifdef _DEBUG
#define EXAFMM_ASSERT(X) assert(X)
#else
#define EXAFMM_ASSERT(X) __assume(X)
#endif

#endif

#endif  // INCLUDE_EXAFMM_PREDEFINES_H_
