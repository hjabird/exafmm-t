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
#if NON_ADAPTIVE
#include "build_non_adaptive_tree.h"
#else
#include "ExaFMM/build_tree.h"
#endif
#include "ExaFMM/build_list.h"
#include "ExaFMM/dataset.h"
#include "ExaFMM/helmholtz.h"

using namespace exafmm_t;

int main(int argc, char **argv) {
  Args args(argc, argv);
  print_divider("Parameters");
  args.print();

  omp_set_num_threads(args.threads);

  print_divider("Time");
  Bodies<complex_t> sources =
      init_sources<complex_t>(args.numBodies, args.distribution, 0);
  Bodies<complex_t> targets =
      init_targets<complex_t>(args.numBodies, args.distribution, 5);

  start("Total");
  complex_t wavek(5, 10);
  HelmholtzFmm fmm(args.P, args.ncrit, wavek);
#if NON_ADAPTIVE
  fmm.depth = args.maxlevel;
#endif

  start("Build Tree");
  get_bounds(sources, targets, fmm.x0, fmm.r0);
  NodePtrs<complex_t> leafs, nonleafs;
  Nodes<complex_t> nodes = build_tree(sources, targets, leafs, nonleafs, fmm);
  stop("Build Tree");

  init_rel_coord();

  start("Build Lists");
  build_list(nodes, fmm);
  stop("Build Lists");

  start("Precomputation");
  fmm.precompute();
  stop("Precomputation");

  start("M2L Setup");
  fmm.M2L_setup(nonleafs);
  stop("M2L Setup");
  fmm.upward_pass(nodes, leafs);
  fmm.downward_pass(nodes, leafs);
  stop("Total");

  bool sample = (args.numBodies >= 10000);
  RealVec err = fmm.verify(leafs, sample);
  print_divider("Error");
  print("Potential Error L2", err[0]);
  print("Gradient Error L2", err[1]);

  print_divider("Tree");
  print("Root Center x", fmm.x0[0]);
  print("Root Center y", fmm.x0[1]);
  print("Root Center z", fmm.x0[2]);
  print("Root Radius R", fmm.r0);
  print("Tree Depth", fmm.depth);
  print("Leaf Nodes", leafs.size());
  print("Helmholtz kD", 2 * fmm.r0 * fmm.wavek);

  return 0;
}
