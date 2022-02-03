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
#include "ExaFMM/laplace.h"

using namespace ExaFMM;

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
  LaplaceFmm fmm(args.P, args.ncrit);
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

  start("Evaluation");
  fmm.upward_pass(nodes, leafs);
  fmm.downward_pass(nodes, leafs);
  stop("Evaluation");

#if DEBUG /* check downward check potential at leaf level*/
  for (auto dn_check : leafs[0]->m_downEquiv) {
    std::cout << dn_check << std::endl;
  }
#endif
  stop("Total");
  print("Evaluation Gflop", (float)m_flop / 1e9);

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
