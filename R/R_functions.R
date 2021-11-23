#' @importFrom Rdpack reprompt
#' @title Desparsified lasso
#' @description Calculates the desparsified lasso as originally introduced in \insertCite{vandeGeer14;textual}{desla}, and provides inference suitable for high-dimensional time series, based on the long run covariance estimator in \insertCite{adamek2020lasso;textual}{desla}.
#' @param X \code{T_}x\code{N} regressor matrix
#' @param y \code{T_}x1 dependent variable vector
#' @param H indexes of relevant regressors
#' @param demean (optional) boolean, true if \code{X} and \code{y} should be demeaned before the desparsified lasso is calculated. This is recommended, due to the assumptions for the method (true by default)
#' @param scale (optional) boolean, true if \code{X} and \code{y} should be scaled by the column-wise standard deviations. Recommended for lasso based methods in general, since the penalty is scale-sensitive (true by default)
#' @param init_partial (optional) boolean, true if you want the initial lasso to be partially penalized (false by default)
#' @param nw_partials (optional) boolean vector with the dimension of \code{H}, trues if you want the nodewise regressions to be partially penalized (all false by default)
#' @param gridsize (optional) integer, how many different lambdas there should be in both initial and nodewise grids (100 by default)
#' @param init_grid (optional) vector, containing user specified initial grid
#' @param nw_grids (optional) matrix with number of rows the size of \code{H}, rows containing user specified grids for the nodewise regressions
#' @param init_selection_type (optional) integer, how should lambda be selected in the initial regression, 1=BIC, 2=AIC, 3=EBIC, 4=PI (4 by default)
#' @param nw_selection_types (optional) integer vector with the dimension of \code{H}, how should lambda be selected in the nodewise regressions, 1=BIC, 2=AIC, 3=EBIC, 4=PI (all 4s by default)
#' @param init_nonzero_limit (optional) number controlling the maximum number of nonzeros that can be selected in the initial regression (0.5 by default, meaning no more than 0.5*T_ regressors can have nonzero estimates)
#' @param nw_nonzero_limits (optional) vector with the dimension of \code{H}, controlling the maximum number of nonzeros that can be selected in the nodewise regressions (0.5s by default)
#' @param init_opt_threshold (optional) optimization threshold for the coordinate descent algorithm in the initial regression (10^(-4) by default)
#' @param nw_opt_thresholds (optional) vector with the dimension of \code{H}, optimization thresholds for the coordinate descent algorithm in the nodewise lasso regression (10^(-4)s by default)
#' @param init_opt_type (optional) integer, which type of coordinate descent algorithm should be used in the initial regression, 1=naive, 2=covariance, 3=adaptive (3 by default)
#' @param nw_opt_types (optional)integer vector with the dimension of \code{H}, which type of coordinate descent algorithm should be used in the nodewise regressions, 1=naive, 2=covariance, 3=adaptive (3s by default)
#' @param LRVtrunc (optional) parameter controlling the bandwidth \code{Q_T} used in the long run covariance matrix, \code{Q_T}=ceil(\code{T_multiplier}*\code{T_}^\code{LRVtrunc}). When \code{LRVtrunc}=\code{T_multiplier}=0, the bandwidth is selected according to \insertCite{andrews1991heteroskedasticity;textual}{desla} (\code{LRVtrunc}=0 by default)
#' @param T_multiplier (optional) parameter controlling the bandwidth \code{Q_T} used in the long run covariance matrix, Q_T=ceil(\code{T_multiplier}*\code{T_}^\code{LRVtrunc}). When \code{LRVtrunc}=\code{T_multiplier}=0, the bandwidth is selected according to \insertCite{andrews1991heteroskedasticity;textual}{desla} (\code{T_multiplier}=0 by default)
#' @param alphas (optional) vector of significance levels (c(0.01,0.05,0.1) by default)
#' @param R (optional) matrix with number of columns the dimension of \code{H}, used to test the null hypothesis \code{R}*beta=\code{q} (identity matrix as default)
#' @param q (optional) vector of size same as the rows of \code{H}, used to test the null hypothesis \code{R}*beta=\code{q} (zeroes by default)
#' @param PIconstant (optional) constant, used in the plug-in selection method (0.8 by default). For details see \insertCite{adamek2020lasso;textual}{desla}
#' @param PIprobability (optional) probability, used in the plug-in selection method (0.05 by default). For details see \insertCite{adamek2020lasso;textual}{desla}
#' @param manual_Thetahat_ (optional) matrix with rows the size of H and columns the number of regressors. Can be obtained from earlier executions of the function to avoid unnecessary calculations of the nodewise regressions (NULL as default)
#' @param manual_Upsilonhat_inv_ (optional) matrix with rows and columns the size of H. Can be obtained from earlier executions of the function to avoid unnecessary calculations of the nodewise regressions (NULL as default)
#' @param manual_nw_residuals_ (optional) matrix with rows equal to the sample size and columns the size of H, containing the residuals from the nodewise regressions. Can be obtained from earlier executions of the function to avoid unnecessary calculations of the nodewise regressions (NULL as default)

#' @return Returns a list with the following elements: \cr
#' \item{\code{bhat_scaled}}{desparsified lasso estimates for the parameters indexed by \code{H}. These estimates are based on data that is potentially standardized, for estimates that are brought back into the original scale of X, see \code{bhat}}
#' \item{\code{bhat}}{desparsified lasso estimates for the parameters indexed by \code{H}, unscaled to be in the original scale of \code{y} and \code{X}}
#' \item{\code{intervals_scaled}}{matrix containing the confidence intervals for parameters indexed in \code{H}, for significance levels given in \code{alphas}. These are based on data that is potentially standardized, for estimates that are brought back into the original scale of X, see \code{intervals}}
#' \item{\code{intervals}}{matrix containing the confidence intervals for parameters indexed in \code{H},unscaled to be in the original scale of \code{y} and \code{X}}
#' \item{\code{joint_chi2_stat}}{test statistic for hull hypothesis \code{R}*beta=\code{q}, asymptotically chi squared distributed}
#' \item{\code{chi2_critical_values}}{critical values of the chi squared distribution with degrees of freedom corresponding to the joint test \code{R}*beta=\code{q}, for significance levels given in \code{alphas}}
#' \item{\code{betahat}}{lasso estimates from the initial regression of \code{y} on \code{X}}
#' \item{\code{Gammahat}}{matrix used for calculating the desparsified lasso, for details see \insertCite{adamek2020lasso;textual}{desla}}
#' \item{\code{Upsilonhat_inv}}{matrix used for calculating the desparsified lasso, for details see \insertCite{adamek2020lasso;textual}{desla}}
#' \item{\code{Thetahat}}{approximate inverse of (X'X)/T_, used for calculating the desparsified lasso, for details see \insertCite{adamek2020lasso;textual}{desla}}
#' \item{\code{Omegahat}}{long run covariance matrix for the variables indexed by \code{H}, for details see \insertCite{adamek2020lasso;textual}{desla}}
#' \item{\code{init_residual}}{vector of residuals from the initial lasso regression}
#' \item{\code{nw_residuals}}{matrix of residuals from the nodewise regressions}
#' \item{\code{init_grid}}{redundant output, returning the function input \code{init_grid}}
#' \item{\code{nw_grids}}{redundant output, returning the function input \code{nw_grids}}
#' \item{\code{init_lambda}}{value of lambda that was selected in the initial lasso regression}
#' \item{\code{nw_lambdas}}{values of lambdas that were selected in the nodewise lasso regressions}
#' \item{\code{init_nonzero}}{number on nonzero parameters in the initial lasso regression}
#' \item{\code{nw_nonzeros}}{vector of nonzero parameters in the nodewise lasso regressions}
#' \item{\code{init_nonzero_pos}}{vector of indexes of the nonzero parameters in the initial lasso}
#' \item{\code{nw_nonzero_poss}}{list of vectors for each nodewise regression, giving the indexes of nonzero parameters in the nodewise regressions}
#' @examples
#' X<-matrix(rnorm(100*100), nrow=100)
#' y<-X[,1:4] %*% c(1, 2, 3, 4) + rnorm(100)
#' H<-c(1, 2, 3, 4)
#' d<-desla(X, y, H)
#' @references
#' \insertAllCited{}
#' @export
desla=function(X, y, H, init_partial=NA, nw_partials=NA, demean=TRUE, scale=TRUE, gridsize=100, init_grid=NA, nw_grids=NA, init_selection_type=NA, nw_selection_types=NA,
                          init_nonzero_limit=NA, nw_nonzero_limits=NA, init_opt_threshold=NA, nw_opt_thresholds=NA, init_opt_type=NA, nw_opt_types=NA,
                          LRVtrunc=0, T_multiplier=0, alphas=c(0.01,0.05,0.1), R=NA, q=NA, PIconstant=0.8, PIprobability=0.05,
                          manual_Thetahat_=NULL, manual_Upsilonhat_inv_=NULL, manual_nw_residuals_=NULL){
  H=H-1 #turns indexes into C++ format
  h=length(H)
  if(!is.matrix(X)){
    if(!is.data.frame(X)){
      warning("X needs to be a matrix or data.frame")
    }else{
      X<-as.matrix(X)
    }
  }
  if(!is.matrix(y)){
    if(!is.data.frame(y) && !is.vector(y)){
      warning("y needs to be a vector, matrix, or data.frame")
    }else{
      y<-as.matrix(y)
    }
  }
  check_cols <- apply(X, 2, function(x){max(x) - min(x) == 0})
  if( (demean || scale) && (sum(check_cols)>0) ){
    warning("constant variable in X, while demean or scale are true, I take demean=scale=FALSE to prevent errors")
    demean<-scale<-FALSE
  }

  if(is.na(init_partial)){
    init_partial=FALSE
  }
  if(is.na(nw_partials[1])){
    nw_partials=rep(FALSE, h)
  }else if(length(nw_partials)==1){
    nw_partials=rep(nw_partials, h)
  }else if(length(nw_partials)!=length(H)){
    warning("length of nw_partials does not match H")
    nw_partials=rep(nw_partials[1], h)
  }

  if(is.na(init_grid) || is.na(nw_grids)){
    g=.Rwrap_build_gridsXy(nrow(X), ncol(X), gridsize, X, y, H, demean, scale)
    if(is.na(init_grid)){
      init_grid=g$init_grid
    }
    if(is.na(nw_grids)){
      nw_grids=g$nw_grids
    }
  }

  if(is.na(init_selection_type)){ #1=BIC, 2=AIC, 3=EBIC, 4=PI
    init_selection_type=4
  }
  if(is.na(nw_selection_types[1])){ #1=BIC, 2=AIC, 3=EBIC, 4=PI
    nw_selection_types=rep(4, h)
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
  if(!is.null(manual_Thetahat_)){
    manual_Thetahat_<-as.matrix(manual_Thetahat_)
    if(nrow(manual_Thetahat_)!=h || ncol(manual_Thetahat_)!=ncol(X)){
      warning(paste0("manual_Thetahat_ has incorrect dimensions"))
    }
  }
  if(!is.null(manual_Upsilonhat_inv_)){
    manual_Upsilonhat_inv_<-as.matrix(manual_Upsilonhat_inv_)
    if(nrow(manual_Upsilonhat_inv_)!=h || ncol(manual_Upsilonhat_inv_)!=h){
      warning(paste0("manual_Upsilonhat_inv_ has incorrect dimensions"))
    }
  }
   if(!is.null(manual_nw_residuals_)){
     manual_nw_residuals_<-as.matrix(manual_nw_residuals_)
     if(nrow(manual_nw_residuals_)!=nrow(X) || ncol(manual_nw_residuals_)!=h){
       warning(paste0("manual_nw_residuals_ has incorrect dimensions"))
     }
   }

  PDLI=.Rwrap_partial_desparsified_lasso_inference(X, y, H, demean, scale, init_partial, nw_partials, init_grid, nw_grids, init_selection_type, nw_selection_types,
                                                  init_nonzero_limit, nw_nonzero_limits, init_opt_threshold, nw_opt_thresholds, init_opt_type, nw_opt_types,
                                                  LRVtrunc, T_multiplier, alphas, R, q, PIconstant, PIprobability,
                                                  manual_Thetahat_, manual_Upsilonhat_inv_, manual_nw_residuals_)
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
  if(!is.null(manual_Thetahat_) && !is.null(manual_Upsilonhat_inv_) && !is.null(manual_nw_residuals_)){ #if all nodewise parts are provided, then the nodewise regressions wont be run
    init_nonzero_pos<-NULL
    nw_nonzero_poss<-NULL
  }else{
    init_nonzero_pos<-PDLI$init$nonzero_pos+1#puts indexes in R format
    nw_nonzero_poss<-PDLI$nw$nonzero_poss
    for(i in 1:length(nw_nonzero_poss)){
      nw_nonzero_poss[[i]]<-nw_nonzero_poss[[i]]+1 #puts the indexes in R format
    }
    names(nw_nonzero_poss)=H
  }
  return(list(bhat_scaled=PDLI$bhat_1,
              bhat=PDLI$bhat_1_unscaled,
              intervals=PDLI$inference$intervals_unscaled,
              intervals_scaled=PDLI$inference$intervals,
              joint_chi2_stat=PDLI$inference$joint_chi2_stat,
              chi2_critical_values=PDLI$inference$chi2_quantiles,
              betahat=PDLI$init$betahat,
              Gammahat=PDLI$Gammahat,
              Upsilonhat_inv=PDLI$Upsilonhat_inv,
              Thetahat=PDLI$Thetahat,
              Omegahat=PDLI$inference$Omegahat,
              init_residual=PDLI$init$residual,
              nw_residuals=PDLI$nw$residuals,
              init_grid=PDLI$init$grid,
              nw_grids=PDLI$nw$grids,
              init_lambda=PDLI$init$lambda,
              nw_lambdas=PDLI$nw$lambdas,
              init_nonzero=PDLI$init$nonzero,
              nw_nonzeros=PDLI$nw$nonzeros,
              init_nonzero_pos=init_nonzero_pos,
              nw_nonzero_poss=nw_nonzero_poss))
}

#' @importFrom Rdpack reprompt
#' @title High-Dimensional Local Projection
#' @description Calculates impulse responses with local projections, using the desla function to estimate the high-dimensional linear models, and provide asymptotic inference. The naming conventions in this function follow the notation in \insertCite{plagborg2021local;textual}{desla}, in particular Equation 1 therein.
#' @param r (optional) vector or matrix with \code{T_} rows, containing the "slow" variables, ones which do not react within the same period to a shock, see \insertCite{plagborg2021local;textual}{desla} for details(NULL by default)
#' @param x \code{T_}x1 vector containing the shock variable, see \insertCite{plagborg2021local;textual}{desla} for details
#' @param y \code{T_}x1 vector containing the response variable, see \insertCite{plagborg2021local;textual}{desla} for details
#' @param q (optional) vector or matrix with \code{T_} rows, containing the "fast" variables, ones which may react within the same period to a shock, see \insertCite{plagborg2021local;textual}{desla} for details (NULL by default)
#' @param manual_w (optional) matrix with \code{T_} rows, containing the variables in w, \insertCite{plagborg2021local;textual}{desla} for details. This overrides the default way it is constructed (NULL by default)
#' @param H (optional) vector of indexes indicating which variables should be left unpenalized (1 by default, which corresponds to the shock variable)
#' @param y_predetermined (optional) boolean, true if the response variable \code{y} is predetermined with respect to \code{x}, i.e. cannot react within the same period to the shock. If true, the impulse response at horizon 0 is 0 (false by default)
#' @param cumulate_y (optional) boolean, true if the impulse response of \code{y} should be cumulated, i.e. using the cumulative sum of \code{y} as the dependent variable (false by default)
#' @param hmax (optional) integer, the maximum horizon up to which the impulse responses are computed. Should not exceed the \code{T_}-\code{lags} (24 by default)
#' @param lags (optional) integer, the number of lags to be included in the local projection model. Should not exceed \code{T_}-\code{hmax}(12 by default)
#' @param alphas (optional) vector of significance levels (0.05 by default)
#' @param init_partial (optional) bool, true if the parameter of interest should NOT be penalized (true by default)
#' @param selection (optional) integer, how should lambda be selected in BOTH the initial and nodewise regressions, 1=BIC, 2=AIC, 3=EBIC, 4=PI (4 by default)
#' @param PIconstant (optional) constant, used in the plug-in selection method (0.8 by default). For details see \insertCite{adamek2020lasso;textual}{desla}
#' @param progress_bar (optional) boolean, true if a progress bar should be displayed during execution (true by default)
#' @return Returns a list with the following elements: \cr
#' \item{\code{intervals}}{matrix containing the point estimates and confidence intervals for the impulse response function, for significance levels given in \code{alphas}}
#' \item{\code{Thetahat}}{matrix (row vector) calculated from the nodewise regression at horizon 0, which is re-used at later horizons}
#' \item{\code{betahats}}{list of matrices (row vectors), giving the initial lasso estimate at each horizon}
#' @examples
#' X<-matrix(rnorm(100*100), nrow=100)
#' y<-X[,1:4] %*% c(1, 2, 3, 4) + rnorm(100)
#' h<-HDLP(x=X[,4], y=y, q=X[,-4], hmax=5, lags=1)
#' @references
#' \insertAllCited{}
#' @export
HDLP=function(r=NULL, x, y, q=NULL, manual_w=NULL, H=1,
                          y_predetermined=FALSE,cumulate_y=FALSE, hmax=24,
                          lags=12, alphas=0.05, init_partial=TRUE, selection=4, PIconstant=0.8,
                          progress_bar=TRUE){
  if(!is.matrix(x)){
    x<-as.matrix(x, ncol=1)
  }
  if(!is.matrix(y)){
    y<-as.matrix(y, ncol=1)
  }
  if(!is.null(r) && !is.matrix(r)){
    r<-as.matrix(r, nrow=nrow(x))
  }
  if(!is.null(q) && !is.matrix(q)){
    q<-as.matrix(q, nrow=nrow(x))
  }
  if(!is.null(manual_w) && !is.matrix(manual_w)){
    manual_w<-as.matrix(manual_w, nrow=nrow(x))
  }
  H<-H-1 #convert to C++ indexing
  if(!is.matrix(alphas)){
    alphas<-as.matrix(alphas, ncol=1)
  }

  LP=.Rcpp_local_projection(r, x, y, q, manual_w, H,
                            y_predetermined,cumulate_y, hmax,
                            lags,alphas, init_partial, selection, PIconstant,
                            progress_bar)
  CInames=rep("",2*length(alphas)+1)
  CInames[length(alphas)+1]="bhat"
  for(i in 1:length(alphas)){
    CInames[i]=paste("lower ", alphas[i], sep="")
    CInames[2*length(alphas)+2-i]=paste("upper ", alphas[i], sep="")
  }
  dimnames(LP$intervals)<-list(horizon=0:hmax, CInames)
  return(list(intervals=LP$intervals,
              Thetahat=LP$manual_Thetahat,
              betahats=LP$betahats))
}
