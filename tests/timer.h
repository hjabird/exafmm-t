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
#ifndef INCLUDE_MFMM_TIMER_H_
#define INCLUDE_MFMM_TIMER_H_

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

namespace mfmm {

static const int stringLength = 20;  //!< Length of formatted string
static const int decimal = 7;        //!< Decimal precision
static const int wait =
    100;  //!< Waiting time between output of different ranks
static const int dividerLength =
    stringLength + decimal + 9;  // length of output section divider
std::atomic<long long> m_flop{0};
static auto m_clock = std::chrono::high_resolution_clock{};
using time_point_t = std::chrono::time_point<decltype(m_clock)>;
time_point_t m_time;
std::map<std::string, time_point_t> m_timer;

void print(std::string s) {
  // if (!VERBOSE | (MPIRANK != 0)) return;
  s += " ";
  std::cout << "--- " << std::setw(stringLength) << std::left
            << std::setfill('-') << s << std::setw(decimal + 1) << "-"
            << std::setfill(' ') << std::endl;
}

template <typename T>
void print(std::string s, T v, bool fixed = true) {
  std::cout << std::setw(stringLength) << std::left << s << " : ";
  if (fixed)
    std::cout << std::setprecision(decimal) << std::fixed << std::scientific;
  else
    std::cout << std::setprecision(1) << std::scientific;
  std::cout << v << std::endl;
}

template <>
void print<std::chrono::duration<double>>(std::string s,
                                          std::chrono::duration<double> v,
                                          bool fixed) {
  std::cout << std::setw(stringLength) << std::left << s << " : ";
  if (fixed)
    std::cout << std::setprecision(decimal) << std::fixed << std::scientific;
  else
    std::cout << std::setprecision(1) << std::scientific;
  std::cout << v.count() << std::endl;
}

void print_divider(std::string s) {
  s.insert(0, " ");
  s.append(" ");
  size_t halfLength = (dividerLength - s.length()) / 2;
  std::cout << std::string(halfLength, '-') << s
            << std::string(dividerLength - halfLength - s.length(), '-')
            << std::endl;
}

void add_flop(long long n) { m_flop += n; }

void start(std::string event) {
  m_time = m_clock.now();
  m_timer[event] = m_time;
}

std::chrono::duration<double> stop(std::string event, bool verbose = true) {
  m_time = m_clock.now();
  std::chrono::duration<double> eventTime = m_time - m_timer[event];
  if (verbose) print(event, eventTime);
  return eventTime;
}
}  // namespace mfmm
#endif  // INCLUDE_MFMM_TIMER_H_
