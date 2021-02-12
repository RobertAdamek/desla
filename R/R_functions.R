#' @importFrom Rdpack reprompt
#' @title Desparsified lasso
#' @description Calculates the desparsified lasso as originally introduced in \insertCite{vandeGeer14;textual}{desla}, and provides inference suitable for high-dimensional time series, based on the long run covariance estimator in \insertCite{adamek2020lasso;textual}{desla}.
#' @param X \code{T}x\code{N} regressor matrix
#' @param y \code{T}x1 dependent variable vector
#' @param H indexes of relevant regressors
#' @param demean (optional) boolean, true if \code{X} and \code{y} should be demeaned before the desparsified lasso is calculated. This is recommended, due to the assumptions for the method (true by default)
#' @param scale (optional) boolean, true if \code{X} and \code{y} should be scaled by the column-wise standard deviations. Recommended for lasso based methods in general, since the penalty is scale-sensitive (true by default)
#' @param init_partial (optional) boolean, true if you want the initial lasso to be partially penalized (false by default)
#' @param nw_partials (optional) boolean vector with the dimension of \code{H}, trues if you want the nodewise regressions to be partially penalized (all false by default)
#' @param gridsize (optional) integer, how many different lambdas there should be in both inital and nodewise grids (100 by default)
#' @param init_grid (optional) vector, containing user specified initial grid
#' @param nw_grids (optional) matrix with number of rows the size of \code{H}, rows containing user specified grids for the nodewise regressions
#' @param init_selection_type (optional) integer, how should lambda be selected in the inital regression, 1=BIC, 2=AIC, 3=EBIC (1 by default)
#' @param nw_selection_types (optional) inteter vector with the dimension of \code{H}, how should lambda be selected in the nodewise regressions, 1=BIC, 2=AIC, 3=EBIC (all 1s by default)
#' @param init_nonzero_limit (optional) number controlling the maximum number of nonzeros that can be selected in the initial regression (0.5 by default, meaning no more than 0.5*T regressors can have nonzero estimates)
#' @param nw_nonzero_limits (optional) vector with the dimension of \code{H}, controlling the maximum number of nonzeros that can be selected in the nodewise regressions (0.5s by default)
#' @param init_opt_threshold (optional) optimization threshold for the coordinate descent algorithm in the inital regression (10^(-4) by default)
#' @param nw_opt_thresholds (optional) vector with the dimension of \code{H}, optimization thresholds for the coordinate descent algorithm in the nodewise lasso regression (10^(-4)s by default)
#' @param init_opt_type (optional) integer, which type of coordinate descent algorithm should be used in the inial regression, 1=naive, 2=covariance, 3=adaptive (3 by default)
#' @param nw_opt_types (optional)inteter vector with the dimension of \code{H}, which type of coordinate descent algorithm should be used in the nodewise regressions, 1=naive, 2=covariance, 3=adaptive (3s by default)
#' @param LRVtrunc (optional) parameter controlling the bandwidth \code{Q_T} used in the long run covariance matrix, \code{Q_T}=ceil(\code{T_multiplier}*\code{T}^\code{LRVtrunc}) (\code{LRVtrunc}=0.2 by default)
#' @param T_multiplier (optional) parameter controlling the bandwidth \code{Q_T} used in the long run covariance matrix, Q_T=ceil(\code{T_multiplier}*\code{T}^\code{LRVtrunc}) (\code{Tmultiplier}=2 by default)
#' @param alphas (optional) vector of significance levels (c(0.01,0.05,0.1) by default)
#' @param R (optional) matrix with number of columns the dimension of \code{H}, used to test the null hypothesis \code{R}*beta=\code{q} (identity matrix as default)
#' @param q (optional) vector of size same as the rows of \code{H}, used to test the null hypothesis \code{R}*beta=\code{q} (zeroes by default)
#' @return Returns a list with the following elements: \cr
#' \item{\code{bhat}}{desparsified lasso estimates for the parameters indexed by \code{H}. These estimates are based on data that is potentially standardized, for estimates that are brought back into the original scale of X, see \code{bhat_unscaled}}
#' \item{\code{bhat_unscaled}}{desparsified lasso estimates for the parameters indexed by \code{H}, unscaled to be in the original scale of \code{y} and \code{X}}
#' \item{\code{intervals}}{matrix containing the confidence intervals for parameters indexed in \code{H}, for significance levels given in \code{alphas}. These are based on data that is potentially standardized, for estimates that are brought back into the original scale of X, see \code{intervals_unscaled}}
#' \item{\code{intervals_unscaled}}{matrix containing the confidence intervals for parameters indexed in \code{H},unscaled to be in the original scale of \code{y} and \code{X}}
#' \item{\code{joint_chi2_stat}}{test statistic for hull hypothesis \code{R}*beta=\code{q}, asymptotically chi squared distributed}
#' \item{\code{chi2_critical_values}}{critical values of the chi squared distribution with degrees of freedom corresponding to the joint test \code{R}*beta=\code{q}, for significance levels given in \code{alphas}}
#' \item{\code{betahat}}{lasso estimates from the inital regression of \code{y} on \code{X}}
#' \item{\code{Gammahat}}{matrix used for calculating the desparsified lasso, for details see \insertCite{adamek2020lasso;textual}{desla}}
#' \item{\code{Upsilonhat_inv}}{matrix used for calculating the desparsified lasso, for details see \insertCite{adamek2020lasso;textual}{desla}}
#' \item{\code{Thetahat}}{approximate inverse of (X'X)/T, used for calculating the desparsified lasso, for details see \insertCite{adamek2020lasso;textual}{desla}}
#' \item{\code{Omegahat}}{long run covariance matrix for the variables indexed by \code{H}, for details see \insertCite{adamek2020lasso;textual}{desla}}
#' \item{\code{init_grid}}{redundant output, returning the function input \code{init_grid}}
#' \item{\code{nw_grids}}{redundant output, returning the function input \code{nw_grids}}
#' \item{\code{init_lambda}}{value of lambda that was selected in the inital lasso regression}
#' \item{\code{nw_lambdas}}{values of lambdas that were selected in the nodewise lasso regressions}
#' \item{\code{init_nonzero}}{redundant output, returning the function input \code{init_nonzero}}
#' \item{\code{nw_nonzeros}}{redundant output, returning the function input \code{nw_nonzeros}}
#' @examples
#' X<-matrix(rnorm(100*100), nrow=100)
#' y<-X[,1:4]%*%c(1,2,3,4)+rnorm(100)
#' H<-c(1, 2, 3, 4)
#' d<-desla(X, y, H)
#' @references
#' \insertAllCited{}
#' @export
desla=function(X, y, H, init_partial=NA, nw_partials=NA, demean=T, scale=T, gridsize=100, init_grid=NA, nw_grids=NA, init_selection_type=NA, nw_selection_types=NA,
                          init_nonzero_limit=NA, nw_nonzero_limits=NA, init_opt_threshold=NA, nw_opt_thresholds=NA, init_opt_type=NA, nw_opt_types=NA,
                          LRVtrunc=0.2, T_multiplier=2, alphas=c(0.01,0.05,0.1), R=NA, q=NA){
  H=H-1 #turns indexes into C++ format
  h=length(H)

  check_cols <- apply(X, 2, function(x){max(x) - min(x) == 0})
  if( (demean || scale) && (sum(check_cols)>0) ){
    warning("Constant variable in X, while demean or scale are true. I take demean=scale=FALSE to prevent errors.")
    demean<-scale<-F
  }

  if(is.na(init_partial)){
    init_partial=F
  }
  if(is.na(nw_partials[1])){
    nw_partials=rep(F, h)
  }else if(length(nw_partials)==1){
    nw_partials=rep(nw_partials, h)
  }else if(length(nw_partials)!=length(H)){
    warning("length of nw_partials does not match H")
    nw_partials=rep(nw_partials[1], h)
  }

  if(is.na(init_grid) || is.na(nw_grids)){
    g=.Rwrap_build_gridsXy(nrow(X), ncol(X), gridsize, X, y, H)
    if(is.na(init_grid)){
      init_grid=g$init_grid
    }
    if(is.na(nw_grids)){
      nw_grids=g$nw_grids
    }
  }

  if(is.na(init_selection_type)){ #1=BIC, 2=AIC, 3=EBIC
    init_selection_type=1
  }
  if(is.na(nw_selection_types[1])){ #1=BIC, 2=AIC, 3=EBIC
    nw_selection_types=rep(1, h)
  }else if(length(nw_selection_types)==1){
    nw_selection_types=rep(nw_selection_types, h)
  }else if(length(nw_selection_types)!=length(H)){
    warning("length of nw_selection_types does not match H")
    nw_selection_types=rep(nw_selection_types[1], h)
  }

  if(is.na(init_nonzero_limit)){
    init_nonzero_limit=0.5
  }
  if(is.na(nw_nonzero_limits[1])){
    nw_nonzero_limits=rep(0.5, h)
  }else if(length(nw_nonzero_limits)==1){
    nw_nonzero_limits=rep(nw_nonzero_limits, h)
  }else if(length(nw_nonzero_limits)!=h){
    warning("length of nw_nonzero_limits does not match H")
    nw_nonzero_limits=rep(nw_nonzero_limits[1], h)
  }

  if(is.na(init_opt_threshold)){
    init_opt_threshold=10^(-4)
  }
  if(is.na(nw_opt_thresholds[1])){
    nw_opt_thresholds=rep(10^(-4), h)
  }else if(length(nw_opt_thresholds)==1){
    nw_opt_thresholds=rep(nw_opt_thresholds, h)
  }else if(length(nw_opt_thresholds)!=h){
    warning("length of nw_opt_thresholds does not match H")
    nw_opt_thresholds=rep(nw_opt_thresholds[1], h)
  }

  if(is.na(init_opt_type)){ #1=naive, 2=covariance, 3=adaptive
    init_opt_type=3
  }
  if(is.na(nw_opt_types[1])){ #1=naive, 2=covariance, 3=adaptive
    nw_opt_types=rep(3, h)
  }else if(length(nw_opt_types)==1){
    nw_opt_types=rep(nw_opt_types, h)
  }else if(length(nw_opt_types)!=h){
    warning("length of nw_opt_types does not match H")
    nw_opt_types=rep(nw_opt_types[1], h)
  }

  alphas=sort(alphas)
  if(is.na(R)){
    R=diag(h)
  }else if(ncol(R)!=h){
    warning("dimensions of R do not match H")
    R=diag(h)
  }
  if(is.na(q)){
    q=rep(0, nrow(R))
  }else if(length(q)!=h){
    warning("length of q does not match H")
    q=rep(0, nrow(R))
  }

  PDLI=.Rwrap_partial_desparsified_lasso_inference(X, y, H, demean, scale, init_partial, nw_partials, init_grid, nw_grids, init_selection_type, nw_selection_types,
                                                  init_nonzero_limit, nw_nonzero_limits, init_opt_threshold, nw_opt_thresholds, init_opt_type, nw_opt_types,
                                                  LRVtrunc, T_multiplier, alphas, R, q)
  CInames=rep("",2*length(alphas)+1)
  CInames[length(alphas)+1]="bhat"
  for(i in 1:length(alphas)){
    CInames[i]=paste("lower ", alphas[i], sep="")
    CInames[2*length(alphas)+2-i]=paste("upper ", alphas[i], sep="")
  }
  colnames(PDLI$inference$intervals)=CInames
  colnames(PDLI$inference$intervals_unscaled)=CInames
  H=H+1 #turns indexes back into R format
  rownames(PDLI$bhat_1)=H
  rownames(PDLI$bhat_1_unscaled)=H
  rownames(PDLI$inference$intervals)=H
  rownames(PDLI$inference$intervals_unscaled)=H
  rownames(PDLI$inference$chi2_quantiles)=alphas
  rownames(PDLI$nw$grids)=H
  rownames(PDLI$nw$lambdas)=H
  rownames(PDLI$nw$nonzeros)=H
  return(list(bhat=PDLI$bhat_1,
              bhat_unscaled=PDLI$bhat_1_unscaled,
              intervals=PDLI$inference$intervals,
              intervals_unscaled=PDLI$inference$intervals_unscaled,
              joint_chi2_stat=PDLI$inference$joint_chi2_stat,
              chi2_critical_values=PDLI$inference$chi2_quantiles,
              betahat=PDLI$init$betahat,
              Gammahat=PDLI$Gammahat,
              Upsilonhat_inv=PDLI$Upsilonhat_inv,
              Thetahat=PDLI$Thetahat,
              Omegahat=PDLI$inference$Omegahat,
              init_grid=PDLI$init$grid,
              nw_grids=PDLI$nw$grids,
              init_lambda=PDLI$init$lambda,
              nw_lambdas=PDLI$nw$lambdas,
              init_nonzero=PDLI$init$nonzero,
              nw_nonzeros=PDLI$nw$nonzeros))
}
