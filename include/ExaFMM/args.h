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

#ifndef INCLUDE_EXAFMM_ARGS_H
#define INCLUDE_EXAFMM_ARGS_H

#include <docopt/docopt.h>

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace ExaFMM {

class Args {
 public:
  int ncrit;
  const char* distribution;
  double k;
  int maxlevel;
  int numBodies;
  int P;
  int threads;

 private:
  inline static const std::string docString{
      R"(Usage: exafmm_program [-cdklnPT]
      
-c <N>, --ncrit <N>             Number of bodies per leaf node to N
-d <type>, --distribution <type>    Cube (c), Sphere (s) or Plummer (p)
-k <k>, --wavenumber <k>        Wavenumber of Helmholtz kernel
-l <N>, --maxlevel <N>          Max level of tree (only applies to non-adaptive tree)
-n <N>, --numBodies <N>         Number of bodies
-P <N>, --P <N>                 Order of expansion
-T <N>, --threads <N>           Number of threads
)"};

  void usage(char* name) {
    fprintf(
        stderr,
        "Usage: %s [options]\n"
        "Long option (short option)     : Description (Default value)\n"
        " --ncrit (-c)                  : Number of bodies per leaf node (%d)\n"
        " --distribution (-d) [c/s/p]   : cube, sphere, plummer (%s)\n"
        " --wavenumber (-k)             : Wavenumber of Helmholtz kernel (%f)\n"
        " --maxlevel (-l)               : Max level of tree (%d) (only applies "
        "to non-adaptive tree)\n"
        " --numBodies (-n)              : Number of bodies (%d)\n"
        " --P (-P)                      : Order of expansion (%d)\n"
        " --threads (-T)                : Number of threads (%d)\n",
        name, ncrit, distribution, k, maxlevel, numBodies, P, threads);
  }

  const char* parseDistribution(const char* arg) {
    switch (arg[0]) {
      case 'c':
        return "cube";
      case 's':
        return "sphere";
      case 'p':
        return "plummer";
      default:
        fprintf(stderr, "invalid distribution %s\n", arg);
        abort();
    }
    return "";
  }

 public:
  Args(int argc = 0, char** argv = nullptr)
      : ncrit(64),
        distribution("cube"),
        k(20),
        maxlevel(5),
        numBodies(100000),
        P(4),
        threads(omp_get_max_threads()) {
    auto settings = docopt::docopt(docString, {argv + 1, argv + argc});
    for (auto& setting : settings) {
      if (setting.first == "ncrit") {
        ncrit = setting.second.asLong();
      } else if (setting.first == "distribution") {
        distribution = parseDistribution(setting.second.asString().data());
      } else if (setting.first == "k") {
        k = std::stod(setting.second.asString());
      } else if (setting.first == "maxlevel") {
        maxlevel = setting.second.asLong();
      } else if (setting.first == "numBodies") {
        numBodies = setting.second.asLong();
      } else if (setting.first == "P") {
        P = setting.second.asLong();
      } else if (setting.first == "threads") {
        threads = setting.second.asLong();
      }
    }
  }

  void print(int stringLength = 20) {
    std::cout << std::setw(stringLength) << std::fixed << std::left << "ncrit"
              << " : " << ncrit << std::endl
              << std::setw(stringLength) << "distribution"
              << " : " << distribution << std::endl
              << std::setw(stringLength) << "numBodies"
              << " : " << numBodies << std::endl
              << std::setw(stringLength) << "P"
              << " : " << this->P << std::endl
              << std::setw(stringLength) << "maxlevel"
              << " : " << maxlevel << std::endl
              << std::setw(stringLength) << "threads"
              << " : " << threads << std::endl
              << std::setw(stringLength) << "wavenumber"
              << " : " << k << std::endl;
  }
};
}  // namespace ExaFMM
#endif  // INCLUDE_EXAFMM_ARGS_H
