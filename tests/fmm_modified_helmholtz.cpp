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
#include <mfmm/build_list.h>
#include <mfmm/build_tree.h>
#include <mfmm/laplace.h>

#include "dataset.h"

using namespace mfmm;

int main(int argc, char **argv) {
  Args args(argc, argv);
  print_divider("Parameters");
  args.print();

  omp_set_num_threads(args.threads);

  print_divider("Time");
  Bodies<real_t> sources =
      init_sources<real_t>(args.numBodies, args.distribution, 0);
  Bodies<real_t> targets =
      init_targets<real_t>(args.numBodies, args.distribution, 5);

  start("Total");
  ModifiedHelmholtzFmm fmm(args.P, args.ncrit, args.k);
#if NON_ADAPTIVE
  fmm.depth = args.maxlevel;
#endif

  start("Build Tree");
  get_bounds(sources, targets, fmm.x0, fmm.r0);
  NodePtrs<real_t> leafs, nonleafs;
  Nodes<real_t> nodes = build_tree(sources, targets, leafs, nonleafs, fmm);
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

  return 0;
}
