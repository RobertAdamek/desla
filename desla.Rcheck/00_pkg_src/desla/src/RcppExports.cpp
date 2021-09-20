// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// Rwrap_partial_desparsified_lasso_inference
List Rwrap_partial_desparsified_lasso_inference(const arma::mat& X, const arma::colvec& y, const arma::uvec& H, const bool& demean, const bool& scale, const bool& init_partial, const LogicalVector& nw_partials, const arma::vec& init_grid, const arma::mat& nw_grids, const int& init_selection_type, const arma::vec& nw_selection_types, const double& init_nonzero_limit, const arma::vec& nw_nonzero_limits, const double& init_opt_threshold, const arma::vec& nw_opt_thresholds, const int& init_opt_type, const arma::vec& nw_opt_types, const double& LRVtrunc, const double& T_multiplier, const NumericVector& alphas, const arma::mat& R, const arma::vec& q);
RcppExport SEXP _desla_Rwrap_partial_desparsified_lasso_inference(SEXP XSEXP, SEXP ySEXP, SEXP HSEXP, SEXP demeanSEXP, SEXP scaleSEXP, SEXP init_partialSEXP, SEXP nw_partialsSEXP, SEXP init_gridSEXP, SEXP nw_gridsSEXP, SEXP init_selection_typeSEXP, SEXP nw_selection_typesSEXP, SEXP init_nonzero_limitSEXP, SEXP nw_nonzero_limitsSEXP, SEXP init_opt_thresholdSEXP, SEXP nw_opt_thresholdsSEXP, SEXP init_opt_typeSEXP, SEXP nw_opt_typesSEXP, SEXP LRVtruncSEXP, SEXP T_multiplierSEXP, SEXP alphasSEXP, SEXP RSEXP, SEXP qSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type H(HSEXP);
    Rcpp::traits::input_parameter< const bool& >::type demean(demeanSEXP);
    Rcpp::traits::input_parameter< const bool& >::type scale(scaleSEXP);
    Rcpp::traits::input_parameter< const bool& >::type init_partial(init_partialSEXP);
    Rcpp::traits::input_parameter< const LogicalVector& >::type nw_partials(nw_partialsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type init_grid(init_gridSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type nw_grids(nw_gridsSEXP);
    Rcpp::traits::input_parameter< const int& >::type init_selection_type(init_selection_typeSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type nw_selection_types(nw_selection_typesSEXP);
    Rcpp::traits::input_parameter< const double& >::type init_nonzero_limit(init_nonzero_limitSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type nw_nonzero_limits(nw_nonzero_limitsSEXP);
    Rcpp::traits::input_parameter< const double& >::type init_opt_threshold(init_opt_thresholdSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type nw_opt_thresholds(nw_opt_thresholdsSEXP);
    Rcpp::traits::input_parameter< const int& >::type init_opt_type(init_opt_typeSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type nw_opt_types(nw_opt_typesSEXP);
    Rcpp::traits::input_parameter< const double& >::type LRVtrunc(LRVtruncSEXP);
    Rcpp::traits::input_parameter< const double& >::type T_multiplier(T_multiplierSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type alphas(alphasSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type R(RSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type q(qSEXP);
    rcpp_result_gen = Rcpp::wrap(Rwrap_partial_desparsified_lasso_inference(X, y, H, demean, scale, init_partial, nw_partials, init_grid, nw_grids, init_selection_type, nw_selection_types, init_nonzero_limit, nw_nonzero_limits, init_opt_threshold, nw_opt_thresholds, init_opt_type, nw_opt_types, LRVtrunc, T_multiplier, alphas, R, q));
    return rcpp_result_gen;
END_RCPP
}
// Rwrap_build_gridsXy
List Rwrap_build_gridsXy(unsigned int& T, unsigned int N, unsigned int& size, arma::mat& X, arma::vec& y, arma::uvec& H);
RcppExport SEXP _desla_Rwrap_build_gridsXy(SEXP TSEXP, SEXP NSEXP, SEXP sizeSEXP, SEXP XSEXP, SEXP ySEXP, SEXP HSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int& >::type T(TSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type N(NSEXP);
    Rcpp::traits::input_parameter< unsigned int& >::type size(sizeSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type H(HSEXP);
    rcpp_result_gen = Rcpp::wrap(Rwrap_build_gridsXy(T, N, size, X, y, H));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_desla_Rwrap_partial_desparsified_lasso_inference", (DL_FUNC) &_desla_Rwrap_partial_desparsified_lasso_inference, 22},
    {"_desla_Rwrap_build_gridsXy", (DL_FUNC) &_desla_Rwrap_build_gridsXy, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_desla(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}