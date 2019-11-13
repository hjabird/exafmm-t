#ifndef modified_helmholtz_h
#define modified_helmholtz_h
#include "exafmm_t.h"
#include "geometry.h"
#include "intrinsics.h"
#include "timer.h"

extern "C" {
  void sgemv_(char* trans, int* m, int* n, float* alpha, float* a, int* lda, float* x,
              int* incx, float* beta, float* y, int* incy);
  void dgemv_(char* trans, int* m, int* n, double* alpha, double* a, int* lda, double* x,
              int* incx, double* beta, double* y, int* incy);
  void sgemm_(char* transa, char* transb, int* m, int* n, int* k, float* alpha, float* a,
              int* lda, float* b, int* ldb, float* beta, float* c, int* ldc);
  void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* a,
              int* lda, double* b, int* ldb, double* beta, double* c, int* ldc);
  void sgesvd_(char *jobu, char *jobvt, int *m, int *n, float *a, int *lda,
               float *s, float *u, int *ldu, float *vt, int *ldvt, float *work, int *lwork, int *info);
  void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *lda,
               double *s, double *u, int *ldu, double *vt, int *ldvt, double *work, int *lwork, int *info);
}

namespace exafmm_t {
  class ModifiedHelmholtzFMM : public FMM {
    using Body_t = Body<real_t>;
    using Bodies_t = Bodies<real_t>;
    using Node_t = Node<real_t>;
    using Nodes_t = Nodes<real_t>;
    using NodePtrs_t = NodePtrs<real_t>;

  public:
    real_t wavek;
    std::vector<RealVec> matrix_UC2E_U;
    std::vector<RealVec> matrix_UC2E_V;
    std::vector<RealVec> matrix_DC2E_U;
    std::vector<RealVec> matrix_DC2E_V;
    std::vector<std::vector<RealVec>> matrix_M2M;
    std::vector<std::vector<RealVec>> matrix_L2L;
    std::vector<M2LData> m2ldata;

    void gemv(int m, int n, real_t* A, real_t* x, real_t* y);
    void gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C);
    void svd(int m, int n, real_t* A, real_t* S, real_t* U, real_t* VT);
    RealVec transpose(RealVec& vec, int m, int n);

    void potential_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value);

    void gradient_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value);

    void kernel_matrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out);

    void initialize_matrix();

    void precompute_check2equiv();

    void precompute_M2M();

    void precompute_M2L(std::ofstream& file, std::vector<std::vector<int>>& parent2child);

    bool load_matrix();

    void save_matrix(std::ofstream& file);

    void precompute();
  };
}  // end namespace exafmm_t
#endif
