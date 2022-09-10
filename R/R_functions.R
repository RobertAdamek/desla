#' @importFrom Rdpack reprompt
#' @title Desparsified lasso
#' @description Calculates the desparsified lasso as originally introduced in \insertCite{vandeGeer14;textual}{desla}, and provides inference suitable for high-dimensional time series, based on the long run covariance estimator in \insertCite{adamek2020lasso;textual}{desla}.
#' @param X \code{T_}x\code{N} regressor matrix
#' @param y \code{T_}x1 dependent variable vector
#' @param H indexes of relevant regressors
#' @param demean (optional) boolean, true if \code{X} and \code{y} should be demeaned before the desparsified lasso is calculated. This is recommended, due to the assumptions for the method (true by default)
#' @param scale (optional) boolean, true if \code{X} and \code{y} should be scaled by the column-wise standard deviations. Recommended for lasso based methods in general, since the penalty is scale-sensitive (true by default)
#' @param init_partial (optional) boolean, true if you want the initial lasso to be partially penalized (false by default)
#' @param nw_partials (optional) boolean vector with the dimension of \code{H}, true if you want the nodewise regressions to be partially penalized (all false by default)
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
#' @param progress_bar (optional) boolean, displays a progress bar while running if true, tracking the progress of estimating the nodewise regressions (TRUE by default)
#' @param parallel boolean, whether parallel computing should be used (TRUE by default)
#' @param threads (optional) integer, how many threads should be used for parallel computing if \code{parallel=TRUE} (default is to use all but two)
#' @param manual_Thetahat_ (optional) matrix with rows the size of H and columns the number of regressors. Can be obtained from earlier executions of the function to avoid unnecessary calculations of the nodewise regressions (NULL as default)
#' @param manual_Upsilonhat_inv_ (optional) matrix with rows and columns the size of H. Can be obtained from earlier executions of the function to avoid unnecessary calculations of the nodewise regressions (NULL by default)
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
#' X<-matrix(rnorm(50*50), nrow=50)
#' y<-X[,1:4] %*% c(1, 2, 3, 4) + rnorm(50)
#' H<-c(1, 2, 3, 4)
#' d<-desla(X, y, H)
#' @references
#' \insertAllCited{}
#' @export
desla=function(X, y, H, init_partial=NA, nw_partials=NA, demean=TRUE, scale=TRUE, gridsize=100, init_grid=NA, nw_grids=NA, init_selection_type=NA, nw_selection_types=NA,
                          init_nonzero_limit=NA, nw_nonzero_limits=NA, init_opt_threshold=NA, nw_opt_thresholds=NA, init_opt_type=NA, nw_opt_types=NA,
                          LRVtrunc=0, T_multiplier=0, alphas=c(0.01,0.05,0.1), R=NA, q=NA, PIconstant=0.8, PIprobability=0.05, progress_bar=TRUE, parallel=TRUE, threads=NULL,
                          manual_Thetahat_=NULL, manual_Upsilonhat_inv_=NULL, manual_nw_residuals_=NULL){
  if (is.numeric(H)) {
    if (!is.null(colnames(X))) {
    } else {
      colnames(X) <- paste0("X", 1:NCOL(X))
    }
    Hnames <- colnames(X)[H]
  } else {
    Hnames <- H
  }
  if(is.data.frame(X)) {
    Xnames <- colnames(X)
    tempXnames <- NULL
    for (i in 1:length(Xnames)) {
      tempXnames <- c(tempXnames, paste0("X", i, Xnames[i]))
    }
    colnames(X) <- tempXnames
    Htemp <- match(Hnames, Xnames)
    for (i in 1:length(Htemp)) {
      Hnames[i] <- paste0("X", Htemp[i], Hnames[i])
    }

    df <- data.frame(y, X)
    if (demean) {
      X <- stats::model.matrix(y ~ ., df)[, -1] # Remove the intercept
    } else {
      X <- stats::model.matrix(y ~ . -1, df) # Do not include the intercept
    }
    Xnames1 <- colnames(X)
    H1 <- NULL
    for (i in 1:length(H)) {
      s <- sapply(1:length(Xnames1), function(x){startsWith(Xnames1[x], tempXnames[H[i]])})
      H1 = c(H1, (1:length(Xnames1))[s])
    }
    for (i in 1:length(Xnames1)) {
      Xnames1[i] <- substr(Xnames1[i], 3, nchar(Xnames1[i]))
    }
    colnames(X) <- Xnames1
    H <- H1
    Hnames <- Xnames1[H]
  }
  if (is.numeric(H)) {
    H=H-1 #turns indexes into C++ format
  } else {
    if (!is.null(colnames(X))) {
      H <- match(H, colnames(X)) - 1
    } else {
      stop("Invalid argument H")
    }
  }
  h=length(H)
  if(!is.matrix(X)){
    # if(!is.data.frame(X)){
    #   warning("X needs to be a matrix or data.frame")
    # }else{
      X<-as.matrix(X)
    # }
  }
  if(!is.matrix(y)){
    # if(!is.data.frame(y) && !is.vector(y)){
    #   warning("y needs to be a vector, matrix, or data.frame")
    # }else{
      y<-as.matrix(y)
    # }
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
  if(parallel){
    if(is.null(threads)){
      threads <- parallelly::availableCores(omit = 2)
    }
  }else{
    threads <- 0
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
                                                  LRVtrunc, T_multiplier, alphas, R, q, PIconstant, PIprobability, progress_bar, threads,
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
  rownames(PDLI$bhat_1_unscaled)=Hnames
  rownames(PDLI$inference$intervals)=Hnames
  rownames(PDLI$inference$intervals_unscaled)=Hnames
  rownames(PDLI$inference$chi2_quantiles)=alphas
  rownames(PDLI$nw$grids)=Hnames
  rownames(PDLI$nw$lambdas)=Hnames
  rownames(PDLI$nw$nonzeros)=Hnames
  if(!is.null(manual_Thetahat_) && !is.null(manual_Upsilonhat_inv_) && !is.null(manual_nw_residuals_)){ #if all nodewise parts are provided, then the nodewise regressions wont be run
    init_nonzero_pos<-NULL
    nw_nonzero_poss<-NULL
  }else{
    init_nonzero_pos<-PDLI$init$nonzero_pos+1#puts indexes in R format
    nw_nonzero_poss<-PDLI$nw$nonzero_poss
    for(i in 1:length(nw_nonzero_poss)){
      nw_nonzero_poss[[i]]<-nw_nonzero_poss[[i]]+1 #puts the indexes in R format
    }
    names(nw_nonzero_poss)=Hnames
  }
  out <- list(bhat_scaled=PDLI$bhat_1,
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
              nw_nonzero_poss=nw_nonzero_poss,
              call = match.call(),
              varnames = colnames(X))
  class(out) <- "desla"
  return(out)
}

#' @importFrom Rdpack reprompt
#' @title State Dependent High-Dimensional Local Projection
#' @description Calculates impulse responses with local projections, using the desla function to estimate the high-dimensional linear models, and provide asymptotic inference. The naming conventions in this function follow the notation in \insertCite{plagborg2021local;textual}{desla}, in particular Equation 1 therein. This function also allows for estimating state-dependent responses, as in \insertCite{ramey2018government;textual}{desla}.
#' @param x \code{T_}x1 vector containing the shock variable, see \insertCite{plagborg2021local;textual}{desla} for details
#' @param y \code{T_}x1 vector containing the response variable, see \insertCite{plagborg2021local;textual}{desla} for details
#' @param r (optional) vector or matrix with \code{T_} rows, containing the "slow" variables, ones which do not react within the same period to a shock, see \insertCite{plagborg2021local;textual}{desla} for details(NULL by default)
#' @param q (optional) vector or matrix with \code{T_} rows, containing the "fast" variables, ones which may react within the same period to a shock, see \insertCite{plagborg2021local;textual}{desla} for details (NULL by default)
#' @param y_predetermined (optional) boolean, true if the response variable \code{y} is predetermined with respect to \code{x}, i.e. cannot react within the same period to the shock. If true, the impulse response at horizon 0 is 0 (false by default)
#' @param cumulate_y (optional) boolean, true if the impulse response of \code{y} should be cumulated, i.e. using the cumulative sum of \code{y} as the dependent variable (false by default)
#' @param hmax (optional) integer, the maximum horizon up to which the impulse responses are computed. Should not exceed the \code{T_}-\code{lags} (24 by default)
#' @param lags (optional) integer, the number of lags to be included in the local projection model. Should not exceed \code{T_}-\code{hmax}(12 by default)
#' @param alphas (optional) vector of significance levels (0.05 by default)
#' @param init_partial (optional) boolean, true if the parameter of interest should NOT be penalized (true by default)
#' @param selection (optional) integer, how should lambda be selected in BOTH the initial and nodewise regressions, 1=BIC, 2=AIC, 3=EBIC, 4=PI (4 by default)
#' @param PIconstant (optional) constant, used in the plug-in selection method (0.8 by default). For details see \insertCite{adamek2020lasso;textual}{desla}
#' @param progress_bar (optional) boolean, true if a progress bar should be displayed during execution (true by default)
#' @param state_variables (optional) matrix or data frame with \code{T_} rows, containing the variables that define the states. Each column should either represent a categorical variable indicating the state of each observation, or each column should be a binary indicator for one particular state; see 'Details'.
#' @param OLS (optional) boolean, whether the local projections should be computed by OLS instead of the desparsified lasso. This should only be done for low-dimensional regressions (FALSE by default)
#' @param parallel boolean, whether parallel computing should be used. Default is TRUE.
#' @param threads (optional) integer, how many threads should be used for parallel computing if \code{parallel=TRUE}. Default is to use all but two.
#' @details The input to \code{state_variables} is transformed to a suitable matrix where each column represents one state using the function \code{\link{create_state_dummies}}. See that function for further details.
#' @return Returns a list with the following elements: \cr
#' \item{\code{intervals}}{list of matrices containing the point estimates and confidence intervals for the impulse response functions in each state, for significance levels given in \code{alphas}}
#' \item{\code{Thetahat}}{matrix (row vector) calculated from the nodewise regression at horizon 0, which is re-used at later horizons}
#' \item{\code{betahats}}{list of matrices (column vectors), giving the initial lasso estimate at each horizon}
#' @examples
#' X<-matrix(rnorm(50*50), nrow=50)
#' y<-X[,1:4] %*% c(1, 2, 3, 4) + rnorm(50)
#' s<-matrix(c(rep(1,25),rep(0,50),rep(1,25)), ncol=2, dimnames = list(NULL, c("A","B")))
#' h<-HDLP(x=X[,4], y=y, q=X[,-4], state_variables=s, hmax=5, lags=1)
#' plot(h)
#' @references
#' \insertAllCited{}
#' @export
HDLP=function(x, y, r=NULL, q=NULL, state_variables=NULL,
              y_predetermined=FALSE,cumulate_y=FALSE, hmax=24,
              lags=12, alphas=0.05, init_partial=TRUE, selection=4, PIconstant=0.8,
              progress_bar=TRUE, OLS=FALSE, parallel=TRUE, threads=NULL){
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
  if(!is.matrix(alphas)){
    alphas<-as.matrix(alphas, ncol=1)
  }
  if(!is.null(state_variables)){
      state_variables <- create_state_dummies(state_variables)
  }
  #A check to make sure that hmax+lags isn't too long compared to the effective sample size
  if(!is.null(state_variables)){
    effective_ss<-min(colSums(state_variables))
  }else{
    effective_ss<-nrow(x)
  }
  if(hmax+lags>0.8*effective_ss){
    hmax<-floor(0.8*effective_ss-lags)
    warning("hmax and lags are too large compared to the effective sample size, taking hmax=floor(0.8*effective_ss-lags) to avoid unexpected behavior")
  }
  if(parallel){
    if(is.null(threads)){
      threads <- parallelly::availableCores(omit = 2)
    }
  }else{
    threads <- 0
  }
  LP=.Rcpp_local_projection_state_dependent(r, x, y, q, state_variables,
                            y_predetermined,cumulate_y, hmax,
                            lags,alphas, init_partial, selection, PIconstant,
                            progress_bar, OLS, threads)
  CInames=rep("",2*length(alphas)+1)
  CInames[length(alphas)+1]="bhat"
  for(i in 1:length(alphas)){
    CInames[i]=paste("lower ", alphas[i], sep="")
    CInames[2*length(alphas)+2-i]=paste("upper ", alphas[i], sep="")
  }
  if(is.null(state_variables)){
    dimnames(LP$intervals)<-list(horizon=0:hmax, CInames)
  }else{
    dimnames(LP$intervals)<-list(horizon=0:hmax, CInames,
                                 state = colnames(state_variables))
  }
  #for(i in 1:length(LP$intervals)){
  #  dimnames(LP$intervals[[i]])<-list(horizon=0:hmax, CInames)
  #}
  #if(!is.null(state_variables)){
  #  names(LP$intervals)<-paste0("state ", 1:ncol(state_variables))
  #}
  varnames <- list(x = "X", y = "Y")
  if (!is.null(colnames(x))) {
    varnames$x = colnames(x)
  }
  if (!is.null(colnames(y))) {
    varnames$y = colnames(y)
  }

  out <- list(intervals=LP$intervals,
              Thetahat=LP$manual_Thetahat,
              betahats=LP$betahats,
              regressors=LP$regressors,
              regressors_trimmed=LP$regressors_trimmed,
              varnames=varnames)
  class(out) <- "hdlp"
  return(out)
}

#' Plot Impulse Responses obtained from HDLP.
#' @param x Output of the \code{HDLP()} function.
#' @param y Has no function, included for compatibility with \code{plot.default()}.
#' @param response Name of the response variable (\code{y} in \code{HDLP()}).
#' @param impulse Name of the shock variable (\code{x} in \code{HDLP()}).
#' @param states Optional names of the states (when applicable). If not provided, names
#' will be determined from \code{x}.
#' @param units Units of the response variable (y-axis label).
#' @param title String containing title of the plot; can be used to overwrite default
#' generated based on the names of the \code{response} and \code{impulse} variables.
#' @param ... Other arguments forwarded to plot function (currently inactive).
#' @return A \code{ggplot} object.
#' @export
plot.hdlp <- function(x, y = NULL, response = NULL, impulse = NULL, states = NULL,
                      units = NULL, title = NULL, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Cannot plot as package ggplot2 not installed.")
  }
  n_alphas <- (dim(x$intervals)[2] - 1) / 2
  if (length(dim(x$intervals)) > 2) {
    intervals <- matrix(aperm(x$intervals, c(1, 3, 2)), ncol = 2 * n_alphas + 1)
  } else {
    intervals <- x$intervals
  }
  n_states <- nrow(intervals) / dim(x$intervals)[1]
  colnames(intervals) <- colnames(x$intervals)
  rownames(intervals) <- rep(rownames(x$intervals), n_states)
  alphas <- sapply(1:n_alphas, function(i){as.numeric(substr(colnames(intervals)[i], 7,
                                                             nchar(colnames(intervals)[i])))})
  s_alphas <- sort(1 - alphas)
  if (n_alphas > 1) {
    lower_bounds <- data.frame(t(apply(intervals[, 1:n_alphas], 1, sort,
                                       decreasing = TRUE)))
    upper_bounds <- data.frame(t(apply(intervals[, 1:n_alphas + 1 + n_alphas], 1, sort)))
  } else {
    lower_bounds <- data.frame(intervals[, 1])
    upper_bounds <- data.frame(intervals[, 3])
  }
  colnames(lower_bounds) <- paste(100 * s_alphas, "% CI Lower Bound")
  colnames(upper_bounds) <- paste(100 * s_alphas, "% CI Upper Bound")

  if (n_states == 1) {
    state = rep(NA, nrow(intervals))
  } else {
    if (is.null(states)) {
      states <- dimnames(x$intervals)[[3]]
    }
    state = rep(states, each = dim(x$intervals)[1])
  }

  ir <- data.frame(Horizon = as.numeric(rownames(intervals)),
                   Estimate = intervals[, n_alphas + 1],
                   lower_bounds, upper_bounds, state = state)

  if (is.null(title)) {
    if (is.null(response)) {
      response <- x$varnames$y
    }
    if (is.null(impulse)) {
      impulse <- x$varnames$x
    }
    title_txt <- paste0("Response of ", response, " to a shock in ", impulse)
  } else {
    title_txt <- title
  }
  if (is.null(units)) {
    units <- "Response"
  }

  Horizon <- NULL
  Estimate <- NULL
  g <- ggplot2::ggplot(data = ir, ggplot2::aes(x = Horizon)) +
    ggplot2::geom_hline(yintercept = 0, color = "black", size = 0.5) +
    ggplot2::geom_line(ggplot2::aes(y = Estimate), color = "navyblue", size = 1) +
    ggplot2::xlab("Horizon") +
    ggplot2::ylab(units) +
    ggplot2::ggtitle(title_txt) +
    ggplot2::theme(panel.grid.major = ggplot2::element_blank(),
                   panel.grid.minor = ggplot2::element_blank())
  if (n_states > 1) {
    g <- g + ggplot2::facet_wrap(ggplot2::vars(state))
  }
  for (i in 1:n_alphas) {
    g <- g + ggplot2::geom_ribbon(ggplot2::aes_string(ymin = names(ir)[i + 2],
                                                      ymax = names(ir)[i + 2 + n_alphas]),
                                  fill = "navyblue", alpha = 1/(n_alphas + 1))
  }
  return(g)
}

#' Create State Dummies
#' @description Creates state dummies for use in \code{\link{HDLP}}.
#' @param x Contains the variables that define the states. Each column should either represent a categorical variable indicating the state of each observation, or each column should be a binary indicator for one particular state.
#' @details The function first checks if \code{x} is already in the correct output format by evaluating if each row sums up to one. If this is not the case, each column is treated as a categorical variable for which its unique entries define the states it can take. If \code{x} contains more than one column, interactions between the variables are created. Example, inputting two variables that can take two states each, results in a total of four possible states, and hence the output matrix contains four columns.
#' @return A matrix where each column is a binary indicator for one state.
#' @export
create_state_dummies <- function(x) {
  # Check if x has more than one column
  if (NCOL(x) > 1 ) {
    # Check if x has numerical or logical entries
    if (is.numeric(x) | is.logical(x)){
      # Check if the vectors already indicate the different states, in which case the rows
      # should sum to one.
      if (all(rowSums(x) == rep(1, NROW(x)))) {
        # No further transformations need to be done
        d <- as.matrix(x)
        if (!is.null(colnames(x))) {
          names_d <- colnames(x)
        } else {
          names_d <- paste0("State ", 1:NCOL(x))
        }
      } else {
        # In this case each column is treated as a categorical variable that indicates the
        # states for that particular variable.
        if (is.null(colnames(x))) {
          colnames(x) <- paste0("Var", 1:NCOL(x))
        }
        d <- create_state_dummies_from_datamatrix(x)
      }
    } else {
      # In this case each column is treated as a categorical variable that indicates the
      # states for that particular variable.
      if (is.null(colnames(x))) {
        colnames(x) <- paste0("Var", 1:NCOL(x))
      }
      d <- create_state_dummies_from_datamatrix(x)
    }
  } else if (NCOL(x) == 1) {
    # x is treated as a categorical variable that indicates the states by its unique entries.
    if (is.null(colnames(x))) {
      varname <- deparse(substitute(x))
    } else {
      varname <- colnames(x)
    }
    d <- create_state_dummies_from_vector(x, varname = varname)
  }
  return(d)
}

#' Create State Dummies from Matrix
#' @description Creates state dummies from matrix-like objects
#' @param x Matrix or data frame where each column represents a state variable.
#' @return A matrix where each column is a binary indicator for one state.
#' @keywords internal
create_state_dummies_from_datamatrix <- function(x) {
  # In this case each column is treated as a categorical variable that indicates the
  # states for that particular variable. We then need to create interaction terms.
  # We will now use a formula to create the corresponding matrix with dummies
  nr_states <- list()
  states <- list()
  varnames <- colnames(x)
  for (i in 1:NCOL(x)) {
    # Create variable name
    var_i_name <- paste0("Var", i)
    # If needed, transform variable i to a factor
    factor_i <- factor(x[, i], levels = unique(x[, i]))
    # Create the matrix with dummies for this vector
    assign(var_i_name, stats::model.matrix(~ factor_i - 1, data.frame(factor_i)))
    # Store the states this variable has
    states[[var_i_name]] <- levels(factor_i)
    nr_states[[var_i_name]] <- 1:length(levels(x[, i]))
  }
  # We create a function multiplying columns of each variable to create all interactions
  function_txt <- paste(paste0("Var", 1:NCOL(x), "[, all_states[x, ", 1:NCOL(x), "]]"),
                        collapse = " * ", sep = "")
  # Create a matrix that gives all possible combinations of states
  all_states <- expand.grid(nr_states)
  nr_all_states <- nrow(all_states)
  # Look over all the possible combinations of the states with the function created above
  d <- sapply(1:nr_all_states,
              eval(parse(text = paste0("function(x){", function_txt, "}"))))
  # Create column names
  names_d <- rep("", nr_all_states)
  for (i in 1:nr_all_states) {
    names_d[i] <- paste0(names_d[i], paste0(varnames[1], ":", states[[1]][all_states[i, 1]]))
    for (j in 2:NCOL(x)) {
      names_d[i] <- paste0(names_d[i],"-", paste0(varnames[j], ":", states[[j]][all_states[i, j]]))
    }
  }
  colnames(d) <- names_d
  return(d)
}

#' Create State Dummies from Vector
#' @description Creates state dummies from vectors
#' @param x Vector representing the state variable.
#' @param varname Name of the state variable.
#' @return A matrix where each column is a binary indicator for one state.
#' @keywords internal
create_state_dummies_from_vector <- function(x, varname = "StateVar") {
  # x is treated as a categorical variable that indicates the states by its unique entries.
  # We will create a data frame with a factor and use a formula to create the
  # corresponding matrix with dummies
  if (is.logical(x)) {
    x <- as.character(x)
  }
  if (is.data.frame(x)) {
    x <- x[, 1]
  }
  factor_i <- factor(x, levels = unique(x))
  # Create the matrix with dummies for this vector
  d <- stats::model.matrix(~ factor_i - 1, data.frame(factor_i))
  colnames(d) <- paste0(varname, ":", levels(factor_i))
  return(d)
}

#' Extract coefficients
#' @inheritParams confint.desla
#' @param scaled return the coefficients for the scaled variables (as used in the estimation)
#' if \code{TRUE} or the re-scaled coefficients for the original variables if \code{FALSE}
#' (default).
#' @export
#' @keywords internal
coef.desla <- function(object, scaled = FALSE, ...) {
  if (scaled) {
    cf <- c(object$bhat_scaled)
    coefnames <- rownames(object$bhat_scaled)
  } else {
    cf <- c(object$bhat)
    coefnames <- rownames(object$bhat)
  }
  names(cf) <- coefnames
  cf
}

#' Confidence intervals for desla objects
#' @param object a \code{desla} object.
#' @param parm which parameters is the confidence interval needed for.
#' @param level confidence level(s).
#' @param ... additional arguments (ignored).
#' @export
#' @keywords internal
confint.desla <- function (object, parm, level = 0.95, ...)
{
  cf <- coef.desla(object)
  if (missing(parm)) {
    parm <- names(cf)
  } else if (is.numeric(parm)) {
    parm <- names(cf)[parm]
  }
  p <- c((1 - level)/2, 1 - (1 - rev(level))/2)
  pc_header <- paste0(format(100 * p, trim = TRUE, digits = 3), " %")
  z <- stats::qnorm(p)
  ci <- matrix(NA, nrow = length(parm), ncol = length(p))
  colnames(ci) <- pc_header
  rownames(ci) = parm

  cnames <- colnames(object$intervals)
  lvl <- as.numeric(substr(cnames[1], 7, nchar(cnames[1])))
  se <- c(cf - object$intervals[, 1]) / stats::qnorm(1 - lvl/2)
  names(se) <- names(cf)

  ci[] <- cf[parm] + se[parm] %o% z
  ci
}

#' Summary of desla output
#' @inheritParams confint.desla
#' @export
#' @keywords internal
summary.desla <- function(object, ...) {
  out <- list()
  cf <- coef.desla(object)
  varnames <- names(cf)
  cnames <- colnames(object$intervals)
  lvl <- as.numeric(substr(cnames[1], 7, nchar(cnames[1])))
  se <- c(cf - object$intervals[, 1]) / stats::qnorm(1 - lvl/2)
  t_val <- cf / se
  p_val <- 2 * stats::pnorm(abs(t_val), lower.tail = FALSE)
  out$coefficients <- cbind(Estimate = cf, "Std. Error" = se, "t value" = t_val,
                            "Pr(>|t|)" = p_val)
  rownames(out$coefficients) <- varnames

  out$lambdas <- c(object$init_lambda, object$nw_lambdas)
  names(out$lambdas) <- c("Initial regression", varnames)

  if (length(object$init_nonzero_pos) == 0) {
    varsel <- "none"
  } else {
    varsel <- object$varnames[object$init_nonzero_pos]
  }
  out$selected <- list("Initial regression" = varsel)
  for (i in 1:length(varnames)) {
    if (length(object$nw_nonzero_pos[[i]]) == 0) {
      varsel <- "none"
    } else {
      varsel <- object$varnames[object$nw_nonzero_pos]
    }
    out$selected[[varnames[i]]] <- varsel
  }

  out$call <- object$call
  class(out) <- "summary.desla"
  out
}

#' Print desla summary output
#' @param x a \code{desla} object.
#' @param digits digits.
#' @param signif.stars show stars of significance.
#' @param show_selected upper bound for which to show the names of selected variables in the
#' lasso regressions (default is 10)
#' @inheritParams confint.desla
#' @export
#' @keywords internal
print.summary.desla <- function (x, digits = max(3L, getOption("digits") - 3L),
                                 signif.stars = getOption("show.signif.stars"),
                                 show_selected = 10, ...) {
  cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"), "\n\n", sep = "")

  cat("\nCoefficients:\n")
  stats::printCoefmat(x$coefficients, digits = digits, signif.stars = signif.stars,
                       a.print = "NA", ...)

  lam <- data.frame(x$lambdas)
  colnames(lam) <- NULL
  cat("\nSelected lambdas:\n")
  print(lam, digits = digits)

  vars <- data.frame(selected = rep("", length(x$selected)))
  rownames(vars) <- names(x$selected)
  long <- FALSE
  text_long <- paste0("(Names not shown when more than ",
                      show_selected, " are selected.\n",
                      "Use print() with argument 'show_selected' set equal to desired ",
                      "number of names.)")
  for (i in 1:length(x$selected)) {
    if (length(x$selected[[i]]) > show_selected) {
      long <- TRUE
      vars$selected[i] <- paste0(length(x$selected[[i]]), " variables")
    } else {
      vars$selected[i] <- paste0(x$selected[[i]], collapse = ", ")
    }
  }
  colnames(vars) <- NULL
  cat("\nSelected variables:\n")
  print(vars)
  if (long) {cat(text_long)}
  invisible(x)
}

#' Print desla output
#' @param x a \code{desla} object.
#' @param digits digits.
#' @param signif.stars show stars of significance.
#' @param show_selected upper bound for which to show the names of selected variables in the
#' lasso regressions (default is 10)
#' @inheritParams confint.desla
#' @export
#' @keywords internal
print.desla <- function (x, digits = max(3L, getOption("digits") - 3L),
                         signif.stars = getOption("show.signif.stars"),
                         show_selected = 10, ...) {
  y <- summary(x, ...)
  print(y, show_selected = show_selected, ...)
}
