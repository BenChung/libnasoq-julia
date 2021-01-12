#include "jlcxx/jlcxx.hpp"
#include "cholmod.h"
#include "linear_solver_wrapper.h"
#include <stdio.h>

void print_arrd(double* arr, int n) {
  for (int i = 0; i < n; i++)
    printf("%f ", arr[i]);
}
void print_arri(int* arr, int n) {
  for (int i = 0; i < n; i++)
    printf("%i ", arr[i]);
}


jlcxx::BoxedValue<nasoq::SolverSettings> mklbl(void* Hraw, double* q)
{
  // this will be passed as a SuiteSparse.CHOLMOD.Sparse object from Julia
  auto *H_csc = (cholmod_sparse*) Hraw;
  auto* H = new nasoq::CSC;
  H->nzmax = H_csc->nzmax;
  H->ncol = H_csc->ncol;
  H->nrow = H_csc->nrow;
  H->stype = H_csc->stype;
  H->packed = H_csc->packed;
  auto* parr = new int[H_csc->ncol+1];
  auto* iarr = new int[H_csc->nzmax];
  for (int i = 0; i < H_csc->ncol+1; i++) parr[i] = (int)((long*)H_csc->p)[i];
  for (int i = 0; i < H_csc->nzmax; i++) iarr[i] = (int)((long*)H_csc->i)[i];
  H->p = parr;
  H->i = iarr;
  H->x = (double*)H_csc->x;
  return jlcxx::create<nasoq::SolverSettings>(H, q);
}

void set_ldl_variant(nasoq::SolverSettings& solver, int variant) {
  solver.ldl_variant = variant;
}

void set_req_ref_iter(nasoq::SolverSettings& solver, int req_ref_iter) {
  solver.req_ref_iter = req_ref_iter;
}

void set_solver_mode(nasoq::SolverSettings& solver, int solver_mode) {
  solver.solver_mode = solver_mode;
}

void set_reg_diag(nasoq::SolverSettings& solver, double reg_diag) {
  solver.reg_diag = reg_diag;
}

void solve_only(nasoq::SolverSettings& solver, jlcxx::ArrayRef<double> output) {
  double* int_soln = solver.solve_only();
  for (int i=0; i < solver.A->nrow; i++) {
    output[i] = int_soln[i];
  }
}

void symbolic_analysis(nasoq::SolverSettings& solver) {
  solver.symbolic_analysis();
}

void numerical_factorization(nasoq::SolverSettings& solver) {
  solver.numerical_factorization();
}

void update_factorization(nasoq::SolverSettings& solver) {
  solver.update_factorization();
}

void free_lbl(nasoq::SolverSettings& solver) {
  delete []solver.A->p;
  delete []solver.A->i;
  delete solver.A;
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
  mod.add_type<nasoq::SolverSettings>("LBL");
  mod.method("lbl_fact", &mklbl);
  mod.method("set_ldl_variant", &set_ldl_variant);
  mod.method("set_req_ref_iter", &set_req_ref_iter);
  mod.method("set_solver_mode", &set_solver_mode);
  mod.method("set_reg_diag", &set_reg_diag);
  mod.method("symbolic_analysis", &symbolic_analysis);
  mod.method("numerical_factorization", &numerical_factorization);
  mod.method("update_factorization", &update_factorization);
  mod.method("solve_only", &solve_only);
  mod.method("free_lbl", &free_lbl);
}
