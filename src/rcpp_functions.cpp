#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <sitmo.h>      // SITMO for C++98 & C++11 PPRNG
// [[Rcpp::depends(sitmo)]]
#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::plugins(openmp)]]
#include <progress.hpp>
#include <progress_bar.hpp>
// [[Rcpp::depends(RcppProgress)]]

using namespace Rcpp;
using namespace arma;
using namespace std;

//struct definitions

struct lasso_output{
  unsigned int N, T_, gridsize;
  arma::vec grid, y;
  arma::mat betahats, X;
  int opt_type; //1="naive", 2="covariance", 3="adaptive"
};

struct streamlined_lasso_output{
  arma::mat betahats;
};

struct partial_lasso_output{
  bool partial;
  unsigned int N, T_, gridsize, h;
  arma::vec grid, y;
  arma::uvec H, minusH;
  arma::mat betahats, betahats_1, betahats_2, X, X_1, X_2;
  int opt_type; //1="naive", 2="covariance", 3="adaptive"
};

struct streamlined_partial_lasso_output{
  arma::mat betahats;
};

struct streamlined_lasso_selected_output{
  unsigned int lambda_pos, nonzero;
  double lambda, SSR;
  arma::vec  residual, betahat;
};

struct partial_lasso_selected_output{
  bool partial;
  unsigned int N, T_, gridsize, lambda_pos, nonzero, h;
  double lambda, nonzero_limit, criterion_value, SSR;
  arma::vec grid, y, residual, betahat, betahat_1, betahat_2;
  arma::uvec H, minusH;
  arma::mat X, X_1, X_2;
  int opt_type; //1="naive", 2="covariance", 3="adaptive"
  int selection_type; //1="BIC", 2="AIC", 3="EBIC"
  arma::uvec nonzero_pos;
};

struct streamlined_partial_lasso_selected_output{
  unsigned int lambda_pos, nonzero;
  double lambda, SSR;
  arma::vec residual, betahat, betahat_1;
};

struct selection_output{
  unsigned int lambda_pos, nonzero;
  double lambda, nonzero_limit, criterion_value, SSR;
  arma::vec betahat, residual;
  int selection_type; //1="BIC", 2="AIC", 3="EBIC"
  arma::uvec nonzero_pos;
};

struct streamlined_selection_output{
  unsigned int lambda_pos, nonzero;
  double lambda,  SSR;
  arma::vec betahat, residual;
};

struct partial_desparsified_lasso_output{
  bool init_partial;
  //LogicalVector nw_partials;
  unsigned int N, T_, h;
  unsigned int init_gridsize, init_lambda_pos, init_nonzero;
  double init_lambda, init_nonzero_limit, init_criterion_value, init_SSR;
  arma::vec bhat_1;
  arma::vec init_grid, y, init_residual, betahat, betahat_1, betahat_2;
  arma::vec nw_gridsizes, nw_lambda_poss, nw_nonzeros, nw_lambdas, nw_nonzero_limits, nw_criterion_values, nw_SSRs;
  arma::uvec H, minusH;
  arma::mat X, X_1, X_2;
  arma::mat gammahats, Gammahat, Upsilonhat_inv, Thetahat;
  arma::mat nw_grids, nw_residuals;
  int init_opt_type; //1="naive", 2="covariance", 3="adaptive"
  int init_selection_type; //1="BIC", 2="AIC", 3="EBIC"
  arma::vec nw_opt_types; //1="naive", 2="covariance", 3="adaptive"
  arma::vec nw_selection_types; //1="BIC", 2="AIC", 3="EBIC"
  arma::uvec init_nonzero_pos;
  std::list<arma::uvec> nw_nonzero_poss;
};

struct partial_desparsified_lasso_inference_output{
  bool init_partial;
  //LogicalVector nw_partials;
  unsigned int N, T_, h;
  unsigned int init_gridsize, init_lambda_pos, init_nonzero;
  double init_lambda, init_nonzero_limit, init_criterion_value, init_SSR;
  arma::vec bhat_1;
  arma::vec bhat_1_unscaled;
  arma::vec init_grid, y, init_residual, betahat, betahat_1, betahat_2;
  arma::vec nw_gridsizes, nw_lambda_poss, nw_nonzeros, nw_lambdas, nw_nonzero_limits, nw_criterion_values, nw_SSRs;
  arma::uvec H, minusH;
  arma::mat X, X_1, X_2;
  arma::mat gammahats, Gammahat, Upsilonhat_inv, Thetahat;
  arma::mat nw_grids, nw_residuals;
  int init_opt_type; //1="naive", 2="covariance", 3="adaptive"
  int init_selection_type; //1="BIC", 2="AIC", 3="EBIC"
  arma::vec nw_opt_types; //1="naive", 2="covariance", 3="adaptive"
  arma::vec nw_selection_types; //1="BIC", 2="AIC", 3="EBIC"
  arma::mat Omegahat;
  arma::mat R;
  arma::vec q;
  arma::vec std_errors_individual_vars;
  arma::vec z_quantiles;
  arma::mat intervals;
  arma::mat intervals_unscaled;
  arma::mat intervals_Rq;
  arma::mat intervals_Rq_unscaled;
  arma::vec z_stats_Rq;
  double joint_chi2_stat;
  arma::vec chi2_quantiles;
  arma::uvec init_nonzero_pos;
  std::list<arma::uvec> nw_nonzero_poss;

  arma::mat intervals_EWC;
  arma::mat intervals_unscaled_EWC;
  arma::mat Omegahat_EWC;

  arma::mat intervals_NWfb;
  arma::mat intervals_unscaled_NWfb;
  arma::mat Omegahat_NWfb;
};

struct simulated_data{
  arma::mat X;
  arma::vec y;
  arma::vec true_parameters;
};

struct simulation_output{
  unsigned int h, P, M;
  arma::vec Ns, Ts;
  arma::uvec H;
  cube true_parameters_H;
  arma::vec alphas;
  arma::mat coverages;
  cube upper_intervals, lower_intervals;
  arma::mat interval_widths_mean, interval_widths_median;
  cube init_lambda_poss, nw_lambda_poss;
  cube init_nonzeros, nw_nonzeros;
  cube betahats, bhats;
  arma::mat chi2_rejection_rate;
  //List debug;
};

struct grids_output{
  arma::vec init_grid;
  arma::mat nw_grids;
};

struct standardize_output{
  arma::vec y_scaled;
  arma::mat X_scaled;
  double y_mean;
  arma::vec X_means;
  double y_sd;
  arma::vec X_sds;
};

struct reg_output{
  arma::mat Omegahat;
  arma::vec z_quantiles;
  arma::mat intervals;
  arma::mat intervals_unscaled;
  arma::mat intervals_unscaled_EWC;
  arma::mat intervals_unscaled_NWfb;
  arma::vec betahat;
  arma::vec b_H;
  arma::vec b_H_unscaled;
  arma::mat Thetahat;
};



//auxilliary functions

selection_output selectBIC(const arma::mat& betahats, const arma::mat& X, const arma::vec& y, const arma::vec& grid, const unsigned int& N, const unsigned int& T_, const unsigned int& gridsize, const double& nonzero_limit){
  double c, best_criterion_value=std::numeric_limits<double>::infinity(), best_SSR=0;
  unsigned int lambda_pos=0, nonzero=0, best_nonzero=0;
  arma::vec betahat(N);
  arma::vec residual(T_), best_residual(T_);
  double SSR=0;
  for(unsigned int i=0;i<gridsize;i++){
    betahat=betahats.col(i);
    nonzero=std::count_if(betahat.begin(), betahat.end(), [](double j){return j!=0;});
    residual=y-X*betahat;
    SSR=as_scalar(residual.t()*residual);
    c=T_*log(SSR/double(T_))+log(T_)*nonzero;
    if(c < best_criterion_value && nonzero<=T_*nonzero_limit){
      best_criterion_value=c;
      lambda_pos=i;
      best_nonzero=nonzero;
      best_residual=residual;
      best_SSR=SSR;
    }
  }
  selection_output ret;
  ret.lambda_pos=lambda_pos;
  ret.nonzero=best_nonzero;
  ret.lambda=grid(lambda_pos);
  ret.nonzero_limit=nonzero_limit;
  ret.criterion_value=best_criterion_value;
  ret.SSR=best_SSR;
  ret.betahat=betahats.col(lambda_pos);
  ret.residual=best_residual;
  ret.selection_type=1; //BIC
  ret.nonzero_pos=find(betahats.col(lambda_pos));
  return ret;
}

selection_output selectAIC(const arma::mat& betahats, const arma::mat& X, const arma::vec& y, const arma::vec& grid, const unsigned int& N, const unsigned int& T_, const unsigned int& gridsize, const double& nonzero_limit){
  double c, best_criterion_value=std::numeric_limits<double>::infinity(), best_SSR=0;
  unsigned int lambda_pos=0, nonzero=0, best_nonzero=0;
  arma::vec betahat(N);
  arma::vec residual(T_), best_residual(T_);
  double SSR=0;
  for(unsigned int i=0;i<gridsize;i++){
    betahat=betahats.col(i);
    nonzero=std::count_if(betahat.begin(), betahat.end(), [](double j){return j!=0;});
    residual=y-X*betahat;
    SSR=as_scalar(residual.t()*residual);
    c=T_*log(SSR/double(T_))+2.0*nonzero;
    if(c < best_criterion_value && nonzero<=T_*nonzero_limit){
      best_criterion_value=c;
      lambda_pos=i;
      best_nonzero=nonzero;
      best_residual=residual;
      best_SSR=SSR;
    }
  }
  selection_output ret;
  ret.lambda_pos=lambda_pos;
  ret.nonzero=best_nonzero;
  ret.lambda=grid(lambda_pos);
  ret.nonzero_limit=nonzero_limit;
  ret.criterion_value=best_criterion_value;
  ret.SSR=best_SSR;
  ret.betahat=betahats.col(lambda_pos);
  ret.residual=best_residual;
  ret.selection_type=2; //AIC
  ret.nonzero_pos=find(betahats.col(lambda_pos));
  return ret;
}

selection_output selectEBIC(const arma::mat& betahats, const arma::mat& X, const arma::vec& y, const arma::vec& grid, const unsigned int& N, const unsigned int& T_, const unsigned int& gridsize, const double& nonzero_limit){
  double c, best_criterion_value=std::numeric_limits<double>::infinity(), best_SSR=0;
  unsigned int lambda_pos=0, nonzero=0, best_nonzero=0;
  arma::vec betahat(N);
  arma::vec residual(T_), best_residual(T_);
  double SSR=0;
  double gamma=1.0;
  for(unsigned int i=0;i<gridsize;i++){
    betahat=betahats.col(i);
    nonzero=std::count_if(betahat.begin(), betahat.end(), [](double j){return j!=0;});
    residual=y-X*betahat;
    SSR=as_scalar(residual.t()*residual);
    c=T_*log(SSR/double(T_))+log(T_)*nonzero+2.0*nonzero*gamma*log(N);
    if(c < best_criterion_value && nonzero<=T_*nonzero_limit){
      best_criterion_value=c;
      lambda_pos=i;
      best_nonzero=nonzero;
      best_residual=residual;
      best_SSR=SSR;
    }
  }
  selection_output ret;
  ret.lambda_pos=lambda_pos;
  ret.nonzero=best_nonzero;
  ret.lambda=grid(lambda_pos);
  ret.nonzero_limit=nonzero_limit;
  ret.criterion_value=best_criterion_value;
  ret.SSR=best_SSR;
  ret.betahat=betahats.col(lambda_pos);
  ret.residual=best_residual;
  ret.selection_type=3; //EBIC
  ret.nonzero_pos=find(betahats.col(lambda_pos));
  return ret;
}

standardize_output standardize(const arma::mat& X, const arma::vec& y, const bool& demean, const bool& scale){
  unsigned int N=X.n_cols;
  unsigned int T_=X.n_rows;
  standardize_output ret;
  if(demean || scale){
    ret.y_mean=mean(y);
    ret.X_means=arma::vec(N, fill::zeros);
    for(unsigned int j=0; j<N; j++){
      ret.X_means(j)=mean(X.col(j));
    }
  }
  if(scale){
    ret.X_sds=arma::vec(N, fill::zeros);
    double sum=0;
    for(unsigned int t=0; t<T_; t++){
      sum+=pow(y(t)-ret.y_mean,2);
    }
    ret.y_sd=sqrt(sum/double(T_));
    for(unsigned int j=0; j<N; j++){
      sum=0;
      for(unsigned int t=0; t<T_; t++){
        sum+=pow(X.at(t,j)-ret.X_means(j),2);
      }
      ret.X_sds(j)=sqrt(sum/double(T_));
    }
  }
  ret.y_scaled=arma::vec(T_, fill::zeros);
  ret.X_scaled=arma::mat(T_,N, fill::zeros);
  if(demean && scale){
    for(unsigned int t=0; t<T_; t++){
      ret.y_scaled(t)=(y(t)-ret.y_mean)/ret.y_sd;
      for(unsigned int j=0; j<N; j++){
        ret.X_scaled.at(t,j)=(X.at(t,j)-ret.X_means(j))/ret.X_sds(j);
      }
    }
  }else if(demean && !scale){
    for(unsigned int t=0; t<T_; t++){
      ret.y_scaled(t)=(y(t)-ret.y_mean);
      for(unsigned int j=0; j<N; j++){
        ret.X_scaled.at(t,j)=(X.at(t,j)-ret.X_means(j));
      }
    }
  }else if(!demean && scale){
    for(unsigned int t=0; t<T_; t++){
      ret.y_scaled(t)=y(t)/ret.y_sd;
      for(unsigned int j=0; j<N; j++){
        ret.X_scaled.at(t,j)=X.at(t,j)/ret.X_sds(j);
      }
    }
  }else{
    ret.y_scaled=y;
    ret.X_scaled=X;
  }
  return ret;
}

arma::vec buildgrid(const int& size, const double& lmax, const double& lmin) {
  arma::vec grid(size);
  grid(0)=lmax;
  for(int i=1;i<size;i++){
    grid(i)=grid(i-1)*exp(-(log(lmax)-log(lmin))/(size-1));
  }
  return grid;
}

grids_output build_gridsXy(const unsigned int& T_, const unsigned int N, const unsigned int& size, const arma::mat& X, const arma::vec& y, const arma::uvec& H, const bool& demean, const bool& scale){
  unsigned int h=H.n_elem;
  unsigned int j,i;
  arma::vec X_j;
  arma::uvec index_minusj;
  arma::mat X_minusj;
  standardize_output s=standardize(X, y, demean, scale);
  double lambda_max=max(abs((s.X_scaled).t()*(s.y_scaled))/double(T_));
  arma::vec init_grid=buildgrid(size,lambda_max,1.0/double(T_*10.0));
  arma::mat nw_grids(h, size);
  for(i=0; i<h; i++){
    j=H(i);
    X_j=(s.X_scaled).col(j);
    index_minusj=linspace<arma::uvec>(0,N-1,N); index_minusj.shed_row(j);
    X_minusj=(s.X_scaled).cols(index_minusj);
    lambda_max=max(abs(X_minusj.t()*X_j)/double(T_));
    nw_grids.row(i)=buildgrid(size,lambda_max,1.0/double(T_*10.0)).t();
  }
  grids_output ret;
  ret.init_grid=init_grid;
  ret.nw_grids=nw_grids;
  return ret;
}

double soft_threshold(const double& z, const double& gamma){
  double ret;
  if(std::abs(z)<=gamma){
    ret=0;
  }else if(z-gamma>0){
    ret=z-gamma;
  }else{
    ret=z+gamma;
  }
  return ret;
}

arma::uvec unique_match(arma::uvec& minusj, arma::uvec& Hminusj){
  arma::uvec ret(Hminusj.n_elem);
  unsigned int k,l;
  for(k=0; k<minusj.n_elem; k++){
    for(l=0; l<Hminusj.n_elem; l++){
      if(minusj(k)==Hminusj(l)){
        ret(l)=k;
      }
    }
  }
  return ret;
}

arma::vec q_sim_mvnorm_sparse_chol_shrink(const arma::mat& Sigma, const int& S, const arma::vec& q, const double& M = 0.0001) {
  int N = Sigma.n_rows;
  arma::sp_mat Sigma_sqrt = sp_mat(chol(Sigma + M * arma::eye(arma::size(Sigma))));
  Rcpp::NumericVector draw = Rcpp::rnorm(S * N);
  arma::mat Z = arma::mat(draw.begin(), S, N, false, true);
  //arma::mat Z= randn(S,N);
  arma::mat Y = Z * Sigma_sqrt;
  arma::vec X = arma::max(abs(Y), 1);
  arma::vec quant = arma::quantile(X, q);
  return quant;
}

arma::mat coordinate_descent_naive(const arma::mat& X, const arma::colvec& y, const arma::vec& grid, const double& opt_threshold,
                                   const unsigned int& N, const unsigned int& T_, const unsigned int& gridsize){
  arma::mat betahats(N,gridsize);
  arma::vec betahat(N,fill::zeros);
  arma::vec betahat_old(N,fill::zeros);
  unsigned int g,j,k;
  arma::uvec Active_set; Active_set.reset(); //see if this is necessary
  arma::uvec temp_Active_set;
  unsigned int active_length=0, nonactive_length=N;
  arma::uvec Nonactive_set=linspace<arma::uvec>(0,N-1,N);
  double change=opt_threshold+1;
  arma::vec x_crossprods(N,fill::zeros);
  arma::vec x_j(T_);
  for(j=0;j<N;j++){
    x_j=X.col(j);
    x_crossprods(j)=as_scalar(x_j.t()*x_j);
  }
  arma::vec full_res=y;
  arma::vec partial_res=full_res;
  //loop through lambda grid
  for(g=0;g<gridsize;g++){
    //run through non-active set
    //Nonactive_set=linspace<arma::uvec>(0,N-1,N); Nonactive_set.shed_rows(Active_set);
    nonactive_length=Nonactive_set.n_elem;
    for(k=0; k<nonactive_length;k++){
      j=Nonactive_set(k);
      x_j=X.col(j);
      partial_res=full_res;//+X.col(j)*betahat(j);
      betahat(j)=soft_threshold(as_scalar(x_j.t()*partial_res)/x_crossprods(j), T_*grid(g)/x_crossprods(j));
      if(betahat(j)!=0){ //new variable becomes nonzero
        active_length++;
        temp_Active_set=arma::uvec(active_length);
        temp_Active_set.head_rows(active_length-1)=Active_set;
        temp_Active_set(active_length-1)=j;
        Active_set=sort(temp_Active_set); //sorting seems to make it faster for some reason
        full_res=partial_res-X.col(j)*betahat(j);// only need to update if this is nonzero
      }
    }
    Nonactive_set=linspace<arma::uvec>(0,N-1,N); Nonactive_set.shed_rows(Active_set); //update nonactive set
    change=opt_threshold+1;
    while(change>opt_threshold){
      //for loop through parameters of the active set
      for(k=0;k<active_length;k++){
        j=Active_set(k);
        x_j=X.col(j);
        partial_res=full_res+X.col(j)*betahat(j);
        betahat(j)=soft_threshold(as_scalar(x_j.t()*partial_res)/x_crossprods(j), T_*grid(g)/x_crossprods(j));
        full_res=partial_res-X.col(j)*betahat(j);
      }
      change=sqrt(as_scalar((betahat-betahat_old).t()*(betahat-betahat_old)))/double(N);
      betahat_old=betahat;
    }
    betahats.col(g)=betahat;
  }
  return betahats;
}

arma::mat coordinate_descent_covariance(const arma::mat& X, const arma::colvec& y, const arma::vec& grid, const double& opt_threshold,
                                        const unsigned int& N, const unsigned int& T_, const unsigned int& gridsize){
  arma::mat betahat_mat(N,gridsize);
  arma::vec betahat(N,fill::zeros);
  arma::vec betahat_old(N,fill::zeros);
  unsigned int g,j,k,l,a,b;
  arma::uvec Active_set;
  arma::uvec temp_Active_set;
  unsigned int active_length=0, nonactive_length=N;
  arma::uvec Nonactive_set=linspace<arma::uvec>(0,N-1,N);
  double change=opt_threshold+1;
  arma::mat x_crossprods(N,N,fill::zeros);
  arma::vec x_j(T_);
  for(j=0;j<N;j++){
    x_j=X.col(j);
    x_crossprods(j,j)=as_scalar(x_j.t()*x_j);
  }
  arma::vec xy(N,1);
  for(j=0;j<N;j++){
    x_j=X.col(j);
    xy(j)=as_scalar(x_j.t()*y);
  }
  //loop through lambda grid
  for(g=0;g<gridsize;g++){
    //one run through non-active set
    nonactive_length=Nonactive_set.n_elem;
    for(k=0; k<nonactive_length;k++){
      j=Nonactive_set(k);
      double x_jx_kbeta_k=0;
      for(l=0;l<N;l++){
        if(l!=j && betahat(l)!=0){
          x_jx_kbeta_k+=x_crossprods(j,l)*betahat(l);
        }
      }
      betahat(j)=soft_threshold((xy(j)-x_jx_kbeta_k)/x_crossprods(j,j), T_*grid(g)/x_crossprods(j,j));
      if(betahat(j)!=0){ //new variable becomes nonzero
        active_length++;
        x_j=X.col(j);
        for(l=0; l<N;l++){ //calculate crossproducts
          x_crossprods(j,l)=x_crossprods(l,j)=as_scalar(x_j.t()*X.col(l));
        }
        temp_Active_set=arma::uvec(active_length);
        temp_Active_set.head_rows(active_length-1)=Active_set;
        temp_Active_set(active_length-1)=j;
        Active_set=sort(temp_Active_set); //sorting seems to make it faster for some reason
      }

    }
    Nonactive_set=linspace<arma::uvec>(0,N-1,N); Nonactive_set.shed_rows(Active_set); //update nonactive set
    change=opt_threshold+1;
    //optimize
    while(change>opt_threshold){
      //loop through parameters of the active set
      for(k=0;k<active_length;k++){
        j=Active_set(k);
        x_j=X.col(j);
        double x_jx_kbeta_k=0;
        for(a=0; a<active_length;a++){
          b=Active_set(a);
          if(b!=j){
            x_jx_kbeta_k+=x_crossprods(j,b)*betahat(b);
          }
        }
        betahat(j)=soft_threshold((xy(j)-x_jx_kbeta_k)/x_crossprods(j,j), T_*grid(g)/x_crossprods(j,j));
      }
      change=sqrt(as_scalar((betahat-betahat_old).t()*(betahat-betahat_old)))/N;
      betahat_old=betahat;
    }
    betahat_mat.col(g)=betahat;
  }
  return betahat_mat;
}

double Andrews91_truncation(const mat& What, const unsigned int& T_, const unsigned int& h){
  arma::vec rhos(h), variances(h);
  arma::vec y_T(T_-1), y_Tm1(T_-1), constant(T_-1,fill::ones), residual(T_-1);
  arma::mat X_T(T_-1,2);
  arma::vec beta(2);
  for(unsigned int i=0; i<h; i++){
    y_T=What.submat(1,i,T_-1,i);
    y_Tm1=What.submat(0,i,T_-2,i);
    X_T=join_horiz(constant,y_Tm1);
    beta=inv(X_T.t()*X_T)*X_T.t()*y_T;
    residual=y_T-X_T*beta;
    rhos(i)=beta(1);
    variances(i)=as_scalar(residual.t()*residual)/double(double(T_)-2.0);
  }
  double numerator=0, denominator=0;
  for(unsigned int i=0; i<h; i++){
    numerator+=4*pow(rhos(i),2)*pow(variances(i),2)/double(pow(1-rhos(i),6)*pow(1+rhos(i),2));
    denominator+=pow(variances(i),2)/double(pow(1-rhos(i),4));
  }
  double alphahat1=numerator/double(denominator);
  double S_T=1.1447*pow(alphahat1*double(T_),double(1.0/double(3.0)));
  return S_T;
}

mat LRVestimator_old(const vec& init_residual, const mat& nw_residuals, const unsigned int& N, const unsigned int& T_, const unsigned int& h, const double& LRVtrunc, const double& T_multiplier){
  mat Omegahat(h,h,fill::zeros);
  vec uhat=init_residual;
  mat vhat=nw_residuals;
  mat What(T_,h);
  for(unsigned int a=0;a<h;a++){
    What.col(a)=uhat%vhat.col(a);
  }
  unsigned int Q_T;
  if(LRVtrunc==0 && T_multiplier==0){//If both the T_multiplier and LRVtrunc are 0, do a data-driven choice of tuning parameter
    Q_T=std::ceil(Andrews91_truncation(What, T_, h));
  }else{
    Q_T=std::ceil(pow(T_multiplier*double(T_),LRVtrunc));
  }
  vec Whatbar=mean(What,0).as_col();
  //this is not memory efficient, I don't need all the xi_l^(j,k) at once
  cube xi(h,h,Q_T);
  for(unsigned int a=0;a<h;a++){
    for(unsigned int b=0;b<h;b++){
      for(unsigned int l=0;l<=Q_T-1;l++){
        //I think I need to keep redefining them since their sizes change
        vec W_j=What.submat(l,a,(T_-1),a);
        vec W_kminusl=What.submat(0,b,(T_-l-1),b);
        xi(a,b,l)=as_scalar( (W_j-as_scalar(Whatbar[a])).t()*(W_kminusl-as_scalar(Whatbar[b])) )/(T_-l);
      }
    }
  }
  //build the lower triangular part of Omegahat
  vec kernel(Q_T);
  kernel(0)=0; //sort out the calculation of the terms for l>=1 with a loop, because the first element is treated differently
  if(Q_T>1){
    for(double l=1;l<Q_T;l++){
      kernel(l)=1.0-double(l)/double(Q_T);
    }
  }
  for(unsigned int a=0;a<h;a++){
    for(unsigned int b=0;b<=a;b++){
      vec xijk=xi.tube(a,b);
      vec xikj=xi.tube(b,a);
      Omegahat(b,a)=Omegahat(a,b)=xijk(0)+as_scalar( kernel.t()*(xijk+xikj) );
    }
  }
  return Omegahat;
}

mat LRVestimator(const vec& init_residual, const mat& nw_residuals, const unsigned int& N, const unsigned int& T_, const unsigned int& h, const double& LRVtrunc, const double& T_multiplier){
  mat Omegahat(h,h,fill::zeros);
  vec uhat=init_residual;
  mat vhat=nw_residuals;
  mat What(T_,h);
  mat Xi_ell(h,h);
  for(unsigned int a=0;a<h;a++){
    What.col(a)=uhat%vhat.col(a);
  }
  double Q_T_double;
  int Q_T;
  if(LRVtrunc==0 && T_multiplier==0){//If both the T_multiplier and LRVtrunc are 0, do a data-driven choice of tuning parameter
    Q_T_double=std::ceil(Andrews91_truncation(What, T_, h));
    Q_T= (int) Q_T_double;
  }else{
    Q_T_double=std::ceil(T_multiplier*pow(double(T_),LRVtrunc));
    Q_T= (int) Q_T_double;
  }
  if(Q_T_double>double(T_)/2.0){
    //warning("Q_T is larger than T/2, taking Q_T=ceil(T/2) to prevent unexpected behavior");
    Q_T_double=std::ceil(double(T_)/2.0);
    Q_T= (int) Q_T_double;
  }
  if(Q_T<=1){ //If Q_T is zero or negative, Omegahat is Xi(0)
    Omegahat=(1.0/double(T_))*What.t()*What;
  }else{ //Otherwise
    Omegahat=(1.0/double(T_))*What.t()*What;
    for(int ell=1; ell<=Q_T-1; ell++){
      Xi_ell=(1.0/double(T_-ell))*(What.rows(ell,T_-1)).t()*What.rows(0,T_-ell-1);
      Omegahat+=(1-double(ell)/double(Q_T))*(Xi_ell+Xi_ell.t());
    }
  }
  return(Omegahat);
}

mat LRVestimatorEWC(const vec& init_residual, const mat& nw_residuals, const unsigned int& N, const unsigned int& T_, const unsigned int& h, const double& LRVtrunc, const double& T_multiplier){
  mat Omegahat(h,h,fill::zeros);
  vec uhat=init_residual;
  mat vhat=nw_residuals;
  mat What(T_,h);
  mat Xi_ell(h,h);
  for(unsigned int a=0;a<h;a++){
    What.col(a)=uhat%vhat.col(a);
  }
  double Q_T_double;
  int Q_T;
  Q_T_double=std::ceil(T_multiplier*pow(double(T_),LRVtrunc));
  Q_T= (int) Q_T_double;

  vec Lambda_hat(h, fill::zeros);
  for(int j=1; j<=Q_T; j++){
    for(unsigned int t=1; t<=T_; t++){
      Lambda_hat+=(What.row(t-1)).t()*sqrt(2.0/double(T_))*cos(datum::pi*j*(double(t)-1.0/2.0)/double(T_));
    }
    Omegahat+=Lambda_hat*Lambda_hat.t()/Q_T_double;
  }
  return(Omegahat);
}

arma::vec unscale(const standardize_output s, const arma::vec& beta_H, const arma::uvec& H, const bool& demean, const bool& scale){
  unsigned int h=H.n_elem;
  arma::vec unscaled_beta_H(h, fill::zeros);
  if(scale){
    for(unsigned int i=0; i<h; i++){
      unsigned int j=H(i);
      unscaled_beta_H(i)=(s.y_sd/s.X_sds(j))*beta_H(i);
    }
  }else{
    unscaled_beta_H=beta_H;
  }
  return unscaled_beta_H;
}

arma::mat custom_rnorm_sitmo(const unsigned int& r, const unsigned int& c, const unsigned int& active_seed_,
                             const double& mu = 0, const double& sd = 1) {
  uint32_t active_seed = static_cast<uint32_t>(active_seed_);
  sitmo::prng eng( active_seed );
  unsigned int length;
  if((r*c)%2 == 0){
    length = r*c;
  }else{
    length=r*c+1;
  }
  arma::vec draw(length);
  //arma::vec draw32(length);
  for(unsigned int i=0; i<length/2; i++){
    //turn 32 bit random number to uniform, and Box-Muller transform into gaussian
    double u1 = double(eng())/sitmo::prng::max();
    double u2 = double(eng())/sitmo::prng::max();
    draw(2*i) = sqrt(-2*log(u1))*cos(2*datum::pi*u2);
    draw(2*i+1) = sqrt(-2*log(u1))*sin(2*datum::pi*u2);

  }
  //const arma::mat Z = arma::mat(draw.begin(), r, c, false, true);
  arma::mat Z(r,c);
  unsigned int count=0;
  for(unsigned int i=0; i<r; i++){
    for(unsigned int j=0; j<c; j++){
      Z(i,j)=draw(count);
      count++;
    }
  }
  return (Z);
}

//main functions

lasso_output lasso(const arma::mat& X, const arma::colvec& y, const arma::vec& grid,
                   const double& opt_threshold, const int& opt_type){
  unsigned int T_=X.n_rows;
  unsigned int N=X.n_cols;
  unsigned int gridsize=grid.n_elem;
  arma::mat betahats(N,gridsize);
  switch(opt_type) {
  case 1: //"naive"
    betahats=coordinate_descent_naive(X, y, grid, opt_threshold,
                                      N, T_, gridsize);
    break;
  case 2: //"covariance"
    betahats=coordinate_descent_covariance(X, y, grid, opt_threshold,
                                           N, T_, gridsize);
    break;
  case 3: //"adaptive"
    if(N>T_){
      betahats=coordinate_descent_naive(X, y, grid, opt_threshold,
                                        N, T_, gridsize);
    }
    else{
      betahats=coordinate_descent_covariance(X, y, grid, opt_threshold,
                                             N, T_, gridsize);
    }
    break;
  default:
    //warning("Warning: Invalid opt_type, choosing type 3");
    if(N>T_){
      betahats=coordinate_descent_naive(X, y, grid, opt_threshold,
                                        N, T_, gridsize);
    }
    else{
      betahats=coordinate_descent_covariance(X, y, grid, opt_threshold,
                                             N, T_, gridsize);
    }
  }
  lasso_output ret;
  ret.N=N;
  ret.T_=T_;
  ret.gridsize=gridsize;
  ret.grid=grid;
  ret.y=y;
  ret.betahats=betahats;
  ret.X=X;
  ret.opt_type=opt_type;
  return(ret);
}

partial_lasso_output partial_lasso(const arma::mat& X, const arma::colvec& y, const arma::uvec& H, const bool& partial, const arma::vec& grid,
                                   const double& opt_threshold, const int& opt_type){
  unsigned int T_=X.n_rows;
  unsigned int N=X.n_cols;
  unsigned int gridsize=grid.n_elem;
  unsigned int h=H.n_elem;
  arma::mat betahats(N,gridsize);
  arma::mat betahats_1(h,gridsize);
  arma::mat betahats_2(N-h,gridsize);
  arma::mat X_1=X.cols(H);
  arma::uvec minusH=linspace<arma::uvec>(0,N-1,N); minusH.shed_rows(H);
  arma::mat X_2=X.cols(minusH);
  lasso_output L;
  if(partial==true && h>0){
    arma::mat X1X1inv=inv(X_1.t()*X_1);
    arma::mat M_X1=(mat(T_,T_,fill::eye)-X_1*X1X1inv*X_1.t());
    arma::mat resX_2=M_X1*X_2;
    arma::vec resy=M_X1*y;
    L=lasso(resX_2, resy, grid,
            opt_threshold, opt_type);
    betahats_2=L.betahats;
    arma::uvec i_;
    for(unsigned int i=0; i<gridsize; i++){
      betahats_1.col(i)=X1X1inv*X_1.t()*(y-X_2*betahats_2.col(i));
      i_=i;
      betahats.submat(H,i_)=betahats_1.col(i);
      betahats.submat(minusH,i_)=betahats_2.col(i);
    }
  }else{
    L=lasso(X, y, grid, opt_threshold, opt_type);
    if(h>0){
      betahats_1=L.betahats.rows(H);
      betahats_2=L.betahats.rows(minusH);
      betahats=L.betahats;
    }else{
      betahats_2=betahats=L.betahats;
    }
  }
  partial_lasso_output ret;
  ret.partial=partial;
  ret.N=N;
  ret.T_=T_;
  ret.gridsize=gridsize;
  ret.h=h;
  ret.grid=grid;
  ret.y=y;
  ret.H=H;
  ret.minusH=minusH;
  ret.betahats=betahats;
  ret.betahats_1=betahats_1;
  ret.betahats_2=betahats_2;
  ret.X=X;
  ret.X_1=X_1;
  ret.X_2=X_2;
  ret.opt_type=opt_type;
  return(ret);
}

selection_output selectPI(const mat& betahats, const mat& X, const vec& y, const arma::uvec& H, const bool& partial, const vec& grid, const unsigned int& N, const unsigned int& T_, const unsigned int& gridsize, const double& nonzero_limit, const double& c, const double& alpha){
  unsigned int K=15; //max iterations
  double improvement_threshold=0.01; //if the % change in lambda is less than this, stop iterating
  unsigned int B=1000; //how many simulations are used to estimate the quantiles of the gaussian maximum
  partial_lasso_output PLO;
  double ybar=mean(y);
  arma::vec uhat= y-ybar;
  double lambda_old;
  lambda_old=grid(0);
  double lambda;
  arma::mat LRCovariance;
  arma::vec Gmax(B);
  arma::mat Gs;
  arma::vec Gmeans(N);
  double cutoff;
  arma::vec lambda_as_vec(1);
  //arma::uvec H(1); H(0)=0;
  arma::vec eigval(N);
  arma::mat eigvec(N,N);
  arma::mat sqrt_cov(N,N);
  arma::mat sqrt_diag_eigval(N,N,fill::zeros);
  arma::mat random_gaussians=randn(N,B);
  unsigned int iteration=K;
  //loop where we iterate to find the best lambda
  for(unsigned int k=0; k<K; k++){
    if(c!=0){
      //LRCovariance=LRVestimator(uhat, X, N, T_, N, 0.5, 2); //get an estimate for the long run covariance matrix
      LRCovariance=LRVestimator(uhat, X, N, T_, N, 0, 0);
      eig_sym(eigval, eigvec, LRCovariance);
      for(unsigned int i=0; i<N; i++){
        sqrt_diag_eigval(i,i)=sqrt(max(0.0,eigval(i))); //negative eigenvalues replaced by 0
      }
      sqrt_cov=eigvec*sqrt_diag_eigval;
      Gs=sqrt_cov*random_gaussians; //generate the correlated gaussians
      for(unsigned int b=0; b<B; b++){
        Gmax(b)=max(abs(Gs.col(b)));
      }
      Gmax=sort(Gmax);
      cutoff=Gmax(B*(1-alpha)); //1-alpha quantile of the randomly generated data
      lambda=c*cutoff/double(sqrt(double(T_))); ///not yet multiplying by 4 here
    }else{
      lambda=0;
    }
    if(abs(lambda-lambda_old)/lambda_old<improvement_threshold){ //Check if the improvement is big enough
      iteration=k;
      k=K; //no more loops after this, but finish the rest of this loop
    }
    lambda_as_vec(0)=lambda;
    PLO=partial_lasso(X, y, H, partial, lambda_as_vec, pow(10,-4), 3);
    uhat=y-X*PLO.betahats;
    lambda_old=lambda;
  }
  //finding the equivalent of lambda pos
  unsigned int pos=0;
  for(unsigned int i=0;i<gridsize;i++){
    if(grid(i)<lambda){
      break;
    }else{
      pos++;
    }
  }
  arma::uvec nonzero_pos=find(PLO.betahats);
  unsigned int nonzero=nonzero_pos.n_elem;

  selection_output ret;
  ret.betahat=PLO.betahats;
  ret.criterion_value=iteration;
  ret.lambda=lambda;
  ret.lambda_pos=pos;
  ret.nonzero=nonzero;
  ret.nonzero_limit=0;
  ret.residual=uhat;
  ret.selection_type=4;
  ret.SSR=as_scalar(uhat.t()*uhat);
  ret.nonzero_pos=nonzero_pos;
  return ret;
}

selection_output selectPI_new(const mat& betahats, const mat& X, const vec& y, const arma::uvec& H, const bool& partial, const vec& grid, const unsigned int& N, const unsigned int& T_, const unsigned int& gridsize, const double& nonzero_limit, const double& c, const double& alpha){
  unsigned int K=15; //max iterations
  double improvement_threshold=0.01; //if the % change in lambda is less than this, stop iterating
  unsigned int B=1000; //how many simulations are used to estimate the quantiles of the gaussian maximum
  partial_lasso_output PLO;
  double ybar=mean(y);
  arma::vec uhat= y-ybar;
  double lambda_old;
  lambda_old=grid(0);
  double lambda;
  arma::mat LRCovariance;
  double cutoff;
  arma::vec lambda_as_vec(1);
  arma::vec quantile(1); quantile(0)=1-alpha;
  unsigned int iteration=K;
  //loop where we iterate to find the best lambda
  for(unsigned int k=0; k<K; k++){
    if(c!=0){
      //LRCovariance=LRVestimator(uhat, X, N, T_, N, 0.5, 2);
      LRCovariance=LRVestimator(uhat, X, N, T_, N, 0, 0); //get an estimate for the long run covariance matrix
      cutoff=as_scalar(q_sim_mvnorm_sparse_chol_shrink(LRCovariance, B, quantile, 0.01));
      lambda=c*cutoff/double(sqrt(double(T_))); ///not yet multiplying by 4 here
    }else{
      lambda=0;
    }
    if(abs(lambda-lambda_old)/lambda_old<improvement_threshold){ //Check if the improvement is big enough
      iteration=k;
      k=K; //no more loops after this, but finish the rest of this loop
    }
    lambda_as_vec(0)=lambda;
    PLO=partial_lasso(X, y, H, partial, lambda_as_vec, pow(10,-4), 3);
    uhat=y-X*PLO.betahats;
    lambda_old=lambda;
  }
  //finding the equivalent of lambda pos
  unsigned int pos=0;
  for(unsigned int i=0;i<gridsize;i++){
    if(grid(i)<lambda){
      break;
    }else{
      pos++;
    }
  }
  arma::uvec nonzero_pos=find(PLO.betahats);
  unsigned int nonzero=nonzero_pos.n_elem;

  selection_output ret;
  ret.betahat=PLO.betahats;
  ret.criterion_value=iteration;
  ret.lambda=lambda;
  ret.lambda_pos=pos;
  ret.nonzero=nonzero;
  ret.nonzero_limit=0;
  ret.residual=uhat;
  ret.selection_type=4;
  ret.SSR=as_scalar(uhat.t()*uhat);
  ret.nonzero_pos=nonzero_pos;
  return ret;
}

selection_output selectPI_seed(const mat& betahats, const mat& X, const vec& y, const arma::uvec& H, const bool& partial, const vec& grid, const unsigned int& N, const unsigned int& T_, const unsigned int& gridsize, const double& nonzero_limit, const double& c, const double& alpha, unsigned int active_seed){
  unsigned int K=15; //max iterations
  double improvement_threshold=0.01; //if the % change in lambda is less than this, stop iterating
  unsigned int B=1000; //how many simulations are used to estimate the quantiles of the gaussian maximum
  partial_lasso_output PLO;
  double ybar=mean(y);
  arma::vec uhat= y-ybar;
  double lambda_old;
  lambda_old=grid(0);
  double lambda;
  arma::mat LRCovariance;
  arma::vec Gmax(B);
  arma::mat Gs;
  arma::vec Gmeans(N);
  double cutoff;
  arma::vec lambda_as_vec(1);
  //arma::uvec H(1); H(0)=0;
  arma::vec eigval(N);
  arma::mat eigvec(N,N);
  arma::mat sqrt_cov(N,N);
  arma::mat sqrt_diag_eigval(N,N,fill::zeros);
  //arma::mat random_gaussians=randn(N,B);
  arma::mat random_gaussians=custom_rnorm_sitmo(N, B, active_seed, 0, 1);
  unsigned int iteration=K;
  //loop where we iterate to find the best lambda
  for(unsigned int k=0; k<K; k++){
    if(c!=0){
      //LRCovariance=LRVestimator(uhat, X, N, T_, N, 0.5, 2); //get an estimate for the long run covariance matrix
      LRCovariance=LRVestimator(uhat, X, N, T_, N, 0, 0);
      eig_sym(eigval, eigvec, LRCovariance);
      for(unsigned int i=0; i<N; i++){
        sqrt_diag_eigval(i,i)=sqrt(max(0.0,eigval(i))); //negative eigenvalues replaced by 0
      }
      sqrt_cov=eigvec*sqrt_diag_eigval;
      Gs=sqrt_cov*random_gaussians; //generate the correlated gaussians
      for(unsigned int b=0; b<B; b++){
        Gmax(b)=max(abs(Gs.col(b)));
      }
      Gmax=sort(Gmax);
      cutoff=Gmax(B*(1-alpha)); //1-alpha quantile of the randomly generated data
      lambda=c*cutoff/double(sqrt(double(T_))); ///not yet multiplying by 4 here
    }else{
      lambda=0;
    }
    if(abs(lambda-lambda_old)/lambda_old<improvement_threshold){ //Check if the improvement is big enough
      iteration=k;
      k=K; //no more loops after this, but finish the rest of this loop
    }
    lambda_as_vec(0)=lambda;
    PLO=partial_lasso(X, y, H, partial, lambda_as_vec, pow(10,-4), 3);
    uhat=y-X*PLO.betahats;
    lambda_old=lambda;
  }
  //finding the equivalent of lambda pos
  unsigned int pos=0;
  for(unsigned int i=0;i<gridsize;i++){
    if(grid(i)<lambda){
      break;
    }else{
      pos++;
    }
  }
  arma::uvec nonzero_pos=find(PLO.betahats);
  unsigned int nonzero=nonzero_pos.n_elem;

  selection_output ret;
  ret.betahat=PLO.betahats;
  ret.criterion_value=iteration;
  ret.lambda=lambda;
  ret.lambda_pos=pos;
  ret.nonzero=nonzero;
  ret.nonzero_limit=0;
  ret.residual=uhat;
  ret.selection_type=4;
  ret.SSR=as_scalar(uhat.t()*uhat);
  ret.nonzero_pos=nonzero_pos;
  return ret;
}


partial_lasso_selected_output partial_lasso_selected(const arma::mat& X, const arma::colvec& y, const arma::uvec& H, const bool& partial, const arma::vec& grid, const int& selection_type, const double& nonzero_limit,
                                                     const double& opt_threshold, const int& opt_type, const double& PIconstant, const double& PIprobability, unsigned int active_seed){
  partial_lasso_output PL=partial_lasso(X, y, H, partial, grid, opt_threshold, opt_type);
  selection_output S;
  switch(selection_type) {
  case 1: //"BIC"
    S=selectBIC(PL.betahats, X, y, grid, PL.N, PL.T_, PL.gridsize,nonzero_limit);
    break;
  case 2: //"AIC"
    S=selectAIC(PL.betahats, X, y, grid, PL.N, PL.T_, PL.gridsize,nonzero_limit);
    break;
  case 3: //"EBIC"
    S=selectEBIC(PL.betahats, X, y, grid, PL.N, PL.T_, PL.gridsize,nonzero_limit);
    break;
  case 4: //"PI"
    //S=selectPI(PL.betahats, X, y, H, partial, grid, PL.N, PL.T_, PL.gridsize,nonzero_limit, PIconstant, PIprobability);
    S=selectPI_seed(PL.betahats, X, y, H, partial, grid, PL.N, PL.T_, PL.gridsize,nonzero_limit, PIconstant, PIprobability, active_seed);
    break;
  default:
    //warning("Warning: Invalid selection_type, choosing type 4");
    //S=selectPI(PL.betahats, X, y, H, partial, grid, PL.N, PL.T_, PL.gridsize,nonzero_limit, PIconstant, PIprobability);
    S=selectPI_seed(PL.betahats, X, y, H, partial, grid, PL.N, PL.T_, PL.gridsize,nonzero_limit, PIconstant, PIprobability, active_seed);
  }
  partial_lasso_selected_output ret;
  ret.partial=partial;
  ret.N=PL.N;
  ret.T_=PL.T_;
  ret.gridsize=PL.gridsize;
  ret.lambda_pos=S.lambda_pos;
  ret.nonzero=S.nonzero;
  ret.h=PL.h;
  ret.lambda=S.lambda;
  ret.nonzero_limit=S.nonzero_limit;
  ret.criterion_value=S.criterion_value;
  ret.SSR=S.SSR;
  ret.grid=PL.grid;
  ret.y=PL.y;
  ret.residual=S.residual;
  ret.betahat=S.betahat;
  ret.betahat_1=S.betahat.rows(PL.H);
  ret.betahat_2=S.betahat.rows(PL.minusH);
  ret.H=PL.H;
  ret.minusH=PL.minusH;
  ret.X=PL.X;
  ret.X_1=PL.X_1;
  ret.X_2=PL.X_2;
  ret.opt_type=PL.opt_type;
  ret.selection_type=S.selection_type;
  ret.nonzero_pos=S.nonzero_pos;
  return ret;
}

struct nodewise_manual{
  bool use_manual;
  arma::mat manual_Thetahat;
  arma::mat manual_Upsilonhat_inv;
  arma::mat manual_nw_residuals;
};

partial_desparsified_lasso_output partial_desparsified_lasso(const arma::mat& X, const arma::colvec& y, const arma::uvec& H, const bool& init_partial, const LogicalVector& nw_partials, const arma::vec& init_grid, const arma::mat& nw_grids,
                                                             const int& init_selection_type, const arma::vec& nw_selection_types,const double& init_nonzero_limit, const arma::vec& nw_nonzero_limits,
                                                             const double& init_opt_threshold, const arma::vec& nw_opt_thresholds, const int& init_opt_type, const arma::vec& nw_opt_types, const double& PIconstant, const double& PIprobability, bool progress_bar, unsigned int threads, arma::vec seeds,
                                                             //Nullable<NumericMatrix> manual_Thetahat_, Nullable<NumericMatrix> manual_Upsilonhat_inv_, Nullable<NumericMatrix> manual_nw_residuals_
                                                             nodewise_manual nm){

  unsigned int h=H.n_elem;
  partial_lasso_selected_output init_L=partial_lasso_selected(X, y, H, init_partial, init_grid, init_selection_type, init_nonzero_limit,
                                                              init_opt_threshold, init_opt_type, PIconstant, PIprobability, seeds(h));
  unsigned int N=init_L.N;
  unsigned int T_=init_L.T_;
  unsigned int j, i;
  arma::uvec i_;
  bool nw_partial;
  int nw_selection_type, nw_opt_type;
  double nw_nonzero_limit, nw_opt_threshold;
  double tauhat_j;
  arma::vec bhat_1(h), x_j(T_), gammahat_j, nw_grid;
  arma::vec nw_gridsizes(h), nw_lambdas(h), nw_criterion_values(h), nw_SSRs(h), nw_lambda_poss(h), nw_nonzeros(h);
  arma::uvec minusj, Hminusj, nw_H;
  arma::mat nw_residuals(T_,h), Thetahat(h,N, fill::zeros), Upsilonhat_inv(h,h, fill::zeros), gammahats(N-1,h), Gammahat(h,N, fill::zeros), Xminusj(T_,N-1);
  partial_lasso_selected_output nw_L;
  std::list<arma::uvec> nw_nonzero_poss(h);
  //If we manually provide ALL the necessary nodewise components, we don't need to estimate them any more
  //if(manual_Thetahat_.isNotNull() && manual_Upsilonhat_inv_.isNotNull() && manual_nw_residuals_.isNotNull()){
  //  NumericMatrix temp_manual_Thetahat(manual_Thetahat_);  // conversion from Nullable<NumericMatrix> to NumericMatrix
  //  Thetahat=as<arma::mat>(temp_manual_Thetahat); //conversion from NumericMatrix to arma::mat

  //  NumericMatrix temp_manual_Upsilonhat_inv(manual_Upsilonhat_inv_);   // conversion from Nullable<NumericMatrix> to NumericMatrix
  //  Upsilonhat_inv=as<arma::mat>(temp_manual_Upsilonhat_inv); //conversion from NumericMatrix to arma::mat

  //  NumericMatrix temp_manual_nw_residuals(manual_nw_residuals_); // conversion from Nullable<NumericMatrix> to NumericMatrix
  //  nw_residuals=as<arma::mat>(temp_manual_nw_residuals); //conversion from NumericMatrix to arma::mat
  if(nm.use_manual){
    Thetahat=nm.manual_Thetahat;
    Upsilonhat_inv=nm.manual_Upsilonhat_inv;
    nw_residuals=nm.manual_nw_residuals;
  }else{
    Progress p_nodewise(h,progress_bar);
    if(threads>0){
#ifdef _OPENMP
      omp_set_num_threads(threads);
#endif
# pragma omp parallel for schedule(dynamic)
      for(i=0;i<h;i++){
        if ( ! Progress::check_abort() ) {
          unsigned int j_p=H(i);
          arma::vec x_j_p=X.col(j_p);
          arma::uvec minusj_p=linspace<arma::uvec>(0,N-1,N); minusj_p.shed_row(j_p);
          arma::mat Xminusj_p(T_,N-1); Xminusj_p=X.cols(minusj_p);
          arma::uvec Hminusj_p=H; Hminusj_p.shed_row(i);
          arma::uvec nw_H_p=unique_match(minusj_p, Hminusj_p);
          arma::vec nw_grid_p=(nw_grids.row(i)).t();
          bool nw_partial_p=nw_partials(i);
          int nw_selection_type_p=nw_selection_types(i);
          double nw_nonzero_limit_p=nw_nonzero_limits(i);
          double nw_opt_threshold_p=nw_opt_thresholds(i);
          int nw_opt_type_p=nw_opt_types(i);
          partial_lasso_selected_output nw_L_p=partial_lasso_selected(Xminusj_p, x_j_p, nw_H_p, nw_partial_p, nw_grid_p, nw_selection_type_p, nw_nonzero_limit_p,
                                                                      nw_opt_threshold_p, nw_opt_type_p, PIconstant, PIprobability, seeds(i));
          double tauhat_j_p=nw_L_p.SSR/double(T_)+2*nw_L_p.lambda*sum(abs(nw_L_p.betahat));
          Upsilonhat_inv(i,i)= 1.0/tauhat_j_p;
          Gammahat(i,j_p)= 1;
          arma::uvec i_p; i_p=i;
          Gammahat.submat(i_p,minusj_p)= -(nw_L_p.betahat).t();

          gammahats.col(i)=nw_L_p.betahat;
          nw_gridsizes(i)=nw_L_p.gridsize;
          nw_lambdas(i)=nw_L_p.lambda;
          nw_criterion_values(i)=nw_L_p.criterion_value;
          nw_SSRs(i)=nw_L_p.SSR;
          nw_residuals.col(i)=nw_L_p.residual;
          nw_lambda_poss(i)=nw_L_p.lambda_pos;
          nw_nonzeros(i)=nw_L_p.nonzero;
          //nw_nonzero_poss.push_back(nw_L_p.nonzero_pos);
          //nw_nonzero_poss(i)=nw_L_p.nonzero_pos;
          std::list<arma::uvec>::iterator it_p=nw_nonzero_poss.begin(); std::advance(it_p,i); *it_p=nw_L_p.nonzero_pos;

          p_nodewise.increment();
        }
      }
    }else{
      for(i=0;i<h;i++){
        if ( ! Progress::check_abort() ) {
          j=H(i);
          x_j=X.col(j);
          minusj=linspace<arma::uvec>(0,N-1,N); minusj.shed_row(j);
          Xminusj=X.cols(minusj);
          Hminusj=H; Hminusj.shed_row(i);
          nw_H=unique_match(minusj, Hminusj);
          nw_grid=(nw_grids.row(i)).t();
          nw_partial=nw_partials(i);
          nw_selection_type=nw_selection_types(i);
          nw_nonzero_limit=nw_nonzero_limits(i);
          nw_opt_threshold=nw_opt_thresholds(i);
          nw_opt_type=nw_opt_types(i);
          nw_L=partial_lasso_selected(Xminusj, x_j, nw_H, nw_partial, nw_grid, nw_selection_type, nw_nonzero_limit,
                                      nw_opt_threshold, nw_opt_type, PIconstant, PIprobability, seeds(i));
          tauhat_j=nw_L.SSR/double(T_)+2*nw_L.lambda*sum(abs(nw_L.betahat));
          Upsilonhat_inv(i,i)= 1.0/tauhat_j;
          Gammahat(i,j)= 1;
          i_=i;
          Gammahat.submat(i_,minusj)= -(nw_L.betahat).t();

          gammahats.col(i)=nw_L.betahat;
          nw_gridsizes(i)=nw_L.gridsize;
          nw_lambdas(i)=nw_L.lambda;
          nw_criterion_values(i)=nw_L.criterion_value;
          nw_SSRs(i)=nw_L.SSR;
          nw_residuals.col(i)=nw_L.residual;
          nw_lambda_poss(i)=nw_L.lambda_pos;
          nw_nonzeros(i)=nw_L.nonzero;
          //nw_nonzero_poss.push_back(nw_L.nonzero_pos);
          //nw_nonzero_poss(i)=nw_L.nonzero_pos;
          std::list<arma::uvec>::iterator it=nw_nonzero_poss.begin(); std::advance(it,i); *it=nw_L.nonzero_pos;

          p_nodewise.increment();
        }
      }
    }
    //If only some of the nodewise components were provided, here they overwrite the ones that were just calculated
    //if(manual_Upsilonhat_inv_.isNotNull()){
    //  NumericMatrix temp_manual_Upsilonhat_inv(manual_Upsilonhat_inv_);   // conversion from Nullable<NumericMatrix> to NumericMatrix
    //  Upsilonhat_inv=as<arma::mat>(temp_manual_Upsilonhat_inv); //conversion from NumericMatrix to arma::mat
    //}
    //if(manual_nw_residuals_.isNotNull()){
    //  NumericMatrix temp_manual_nw_residuals(manual_nw_residuals_); // conversion from Nullable<NumericMatrix> to NumericMatrix
    //  nw_residuals=as<arma::mat>(temp_manual_nw_residuals); //conversion from NumericMatrix to arma::mat
    //}
    //if(manual_Thetahat_.isNotNull()){
    //  NumericMatrix temp_manual_Thetahat(manual_Thetahat_);  // conversion from Nullable<NumericMatrix> to NumericMatrix
    //  Thetahat=as<arma::mat>(temp_manual_Thetahat); //conversion from NumericMatrix to arma::mat
    //}else{
    //  Thetahat=Upsilonhat_inv*Gammahat; //Note that this Thetahat can be calculated with the manually provided Upsilonhat_inv
    //}
  }
  Thetahat=Upsilonhat_inv*Gammahat;
  bhat_1=init_L.betahat_1+Thetahat*X.t()*(init_L.residual)/double(T_);

  partial_desparsified_lasso_output ret;
  ret.init_partial=init_partial;
  //ret.nw_partials=nw_partials;
  ret.N=init_L.N;
  ret.T_=init_L.T_;
  ret.h=init_L.h;
  ret.init_gridsize=init_L.gridsize;
  ret.init_lambda_pos=init_L.lambda_pos;
  ret.init_nonzero=init_L.nonzero;
  ret.init_lambda=init_L.lambda;
  ret.init_nonzero_limit=init_nonzero_limit;
  ret.init_criterion_value=init_L.criterion_value;
  ret.init_SSR=init_L.SSR;
  ret.bhat_1=bhat_1;
  ret.init_grid=init_grid;
  ret.y=y;
  ret.init_residual=init_L.residual;
  ret.betahat=init_L.betahat;
  ret.betahat_1=init_L.betahat_1;
  ret.betahat_2=init_L.betahat_2;
  ret.nw_gridsizes=nw_gridsizes;
  ret.nw_lambda_poss=nw_lambda_poss;
  ret.nw_nonzeros=nw_nonzeros;
  ret.nw_lambdas=nw_lambdas;
  ret.nw_nonzero_limits=nw_nonzero_limits;
  ret.nw_criterion_values=nw_criterion_values;
  ret.nw_SSRs=nw_SSRs;
  ret.H=init_L.H;
  ret.minusH=init_L.minusH;
  ret.X=X;
  ret.X_1=init_L.X_1;
  ret.X_2=init_L.X_2;
  ret.gammahats=gammahats;
  ret.Gammahat=Gammahat;
  ret.Upsilonhat_inv=Upsilonhat_inv;
  ret.Thetahat=Thetahat;
  ret.nw_grids=nw_grids;
  ret.nw_residuals=nw_residuals;
  ret.init_opt_type=init_opt_type;
  ret.init_selection_type=init_selection_type;
  ret.nw_opt_types=nw_opt_types;
  ret.nw_selection_types=nw_selection_types;
  ret.init_nonzero_pos=init_L.nonzero_pos;
  ret.nw_nonzero_poss=nw_nonzero_poss;
  return ret;
}

partial_desparsified_lasso_inference_output partial_desparsified_lasso_inference(const arma::mat& X, const arma::colvec& y, const arma::uvec& H, const bool& demean, const bool& scale, const bool& init_partial, const LogicalVector& nw_partials,
                                                                                 const arma::vec& init_grid, const arma::mat& nw_grids, const int& init_selection_type, const arma::vec& nw_selection_types,
                                                                                 const double& init_nonzero_limit, const arma::vec& nw_nonzero_limits, const double& init_opt_threshold, const arma::vec& nw_opt_thresholds, const int& init_opt_type, const arma::vec& nw_opt_types,
                                                                                 const double& LRVtrunc, const double& T_multiplier, const NumericVector& alphas, const arma::mat& R, const arma::vec& q, const double& PIconstant, const double& PIprobability, bool progress_bar, unsigned int threads, arma::vec seeds,
                                                                                 //Nullable<NumericMatrix> manual_Thetahat_, Nullable<NumericMatrix> manual_Upsilonhat_inv_, Nullable<NumericMatrix> manual_nw_residuals_
                                                                                 nodewise_manual nm){
  standardize_output s=standardize(X, y, demean, scale);
  partial_desparsified_lasso_output PDL=partial_desparsified_lasso(s.X_scaled, s.y_scaled, H, init_partial, nw_partials, init_grid, nw_grids, init_selection_type, nw_selection_types,
                                                                   init_nonzero_limit, nw_nonzero_limits, init_opt_threshold, nw_opt_thresholds, init_opt_type, nw_opt_types, PIconstant, PIprobability, progress_bar, threads, seeds,
                                                                   //manual_Thetahat_, manual_Upsilonhat_inv_, manual_nw_residuals_
                                                                   nm);
  arma::vec bhat_1_unscaled=unscale(s, PDL.bhat_1, H, demean, scale);
  arma::mat Omegahat=LRVestimator(PDL.init_residual, PDL.nw_residuals, PDL.N, PDL.T_, PDL.h, LRVtrunc, T_multiplier);
  arma::vec z_quantiles=qnorm(alphas/2.0,0.0,1.0,false,false);
  unsigned int P=R.n_rows;
  arma::vec chi2_quantiles=qchisq(alphas,P,false,false);
  arma::mat covariance=PDL.Upsilonhat_inv*Omegahat*PDL.Upsilonhat_inv;
  arma::vec Rbhat_1=R*PDL.bhat_1;
  arma::vec Rbhat_1_unscaled=R*bhat_1_unscaled;
  arma::vec std_errors_individual_vars(PDL.h);
  for(unsigned int i=0; i<PDL.h; i++){
    std_errors_individual_vars(i)=sqrt( as_scalar(covariance(i,i))/double(PDL.T_) );
  }
  arma::vec std_errors_individual_vars_unscale=unscale(s, std_errors_individual_vars, H, demean, scale);
  arma::vec std_errors(P);
  for(unsigned int p=0; p<P; p++){
    std_errors(p)=sqrt( as_scalar(R.row(p)*covariance*(R.row(p)).t())/double(PDL.T_) );
  }
  arma::mat intervals(PDL.h, 2*alphas.length()+1); // intervals for individual variables
  intervals.col(alphas.length())=PDL.bhat_1;
  for(unsigned int p=0; p<PDL.h; p++){
    for(unsigned int j=0; j<alphas.length(); j++){
      intervals(p,j)=PDL.bhat_1(p)-z_quantiles(j)*std_errors_individual_vars(p);
      intervals(p, 2*alphas.length()-j)=PDL.bhat_1(p)+z_quantiles(j)*std_errors_individual_vars(p);
    }
  }
  arma::mat intervals_unscaled=intervals; //unscaling works with just the normal unscale function
  for(int j=0; j<2*alphas.length()+1; j++){
    intervals_unscaled.col(j)=unscale(s, intervals.col(j), H, demean, scale);
  }
  arma::mat intervals_Rq(P, 2*alphas.length()+1); // intervals for the hypotheses tested with R*beta=q
  intervals_Rq.col(alphas.length())=Rbhat_1;
  for(unsigned int p=0; p<P; p++){
    for(unsigned int j=0; j<alphas.length(); j++){
      intervals_Rq(p,j)=Rbhat_1(p)-z_quantiles(j)*std_errors(p);
      intervals_Rq(p, 2*alphas.length()-j)=Rbhat_1(p)+z_quantiles(j)*std_errors(p);
    }
  }
  arma::mat intervals_Rq_unscaled=intervals_Rq; //the hypothesis intervals need to be unscaled differently
  intervals_Rq_unscaled.col(alphas.length())=Rbhat_1_unscaled;
  for(unsigned int p=0; p<P; p++){
    for(unsigned int j=0; j<alphas.length(); j++){
      intervals_Rq_unscaled(p,j)=Rbhat_1_unscaled(p)-z_quantiles(j)*std_errors(p)*Rbhat_1_unscaled(p)/Rbhat_1(p);
      intervals_Rq_unscaled(p, 2*alphas.length()-j)=Rbhat_1_unscaled(p)+z_quantiles(j)*std_errors(p)*Rbhat_1_unscaled(p)/Rbhat_1(p);
    }
  }
  //For the chi2 statistic, the Rbhat-q component should be calculated with a properly scaled q, such that the scale of Rbhat matches the scale of the hypothesized value
  arma::vec q_scaled(P);
  for(unsigned int p=0; p<P; p++){
    q_scaled(p)=q(p)*Rbhat_1(p)/Rbhat_1_unscaled(p);
  }
  double joint_chi2_stat=as_scalar( (Rbhat_1-q_scaled).t()*inv_sympd(R*covariance*R.t()/double(PDL.T_))*(Rbhat_1-q_scaled) );

  arma::vec z_stats_Rq(P);
  for(unsigned int p=0; p<P; p++){
    z_stats_Rq(p)=(Rbhat_1(p)-q_scaled(p))/std_errors(p);
  }

  ////this part deals with doing inference via the EWC instead of NW
  arma::mat Omegahat_EWC=LRVestimatorEWC(PDL.init_residual, PDL.nw_residuals, PDL.N, PDL.T_, PDL.h, 2/3, 0.4);
  double nu=std::ceil(0.4*pow(double(PDL.T_),2/3));
  arma::vec t_nu_quantiles=qt(alphas/2.0,nu,false,false);
  arma::mat covariance_EWC=PDL.Upsilonhat_inv*Omegahat_EWC*PDL.Upsilonhat_inv;
  arma::vec std_errors_individual_vars_EWC(PDL.h);
  for(unsigned int i=0; i<PDL.h; i++){
    std_errors_individual_vars_EWC(i)=sqrt( as_scalar(covariance_EWC(i,i))/double(PDL.T_) );
  }
  arma::vec std_errors_individual_vars_unscale_EWC=unscale(s, std_errors_individual_vars_EWC, H, demean, scale);
  arma::mat intervals_EWC(PDL.h, 2*alphas.length()+1); // intervals for individual variables
  intervals_EWC.col(alphas.length())=PDL.bhat_1;
  for(unsigned int p=0; p<PDL.h; p++){
    for(unsigned int j=0; j<alphas.length(); j++){
      intervals_EWC(p,j)=PDL.bhat_1(p)-t_nu_quantiles(j)*std_errors_individual_vars_EWC(p);
      intervals_EWC(p, 2*alphas.length()-j)=PDL.bhat_1(p)+t_nu_quantiles(j)*std_errors_individual_vars_EWC(p);
    }
  }
  arma::mat intervals_unscaled_EWC=intervals_EWC; //unscaling works with just the normal unscale function
  for(int j=0; j<2*alphas.length()+1; j++){
    intervals_unscaled_EWC.col(j)=unscale(s, intervals_EWC.col(j), H, demean, scale);
  }
  ////////////////////////////////////////////////////////////////////


  ////this part deals with the fixed-b asymptotics of NW
  arma::mat Omegahat_NWfb=LRVestimator(PDL.init_residual, PDL.nw_residuals, PDL.N, PDL.T_, PDL.h, 1/2, 1.3);
  double S=std::ceil(1.3*pow(double(PDL.T_),1/2));
  double a0=1.96, a1=2.9694, a2=0.3142, a3=-0.3427; //taken from Table 1 of Kiefer and Vogelsang (2005)
  double b=S/PDL.T_;
  double cv=a0+a1*b+a2*pow(b,2)+a3*pow(b,3);
  arma::mat covariance_NWfb=PDL.Upsilonhat_inv*Omegahat_NWfb*PDL.Upsilonhat_inv;
  arma::vec std_errors_individual_vars_NWfb(PDL.h);
  for(unsigned int i=0; i<PDL.h; i++){
    std_errors_individual_vars_NWfb(i)=sqrt( as_scalar(covariance_NWfb(i,i))/double(PDL.T_) );
  }
  arma::vec std_errors_individual_vars_unscale_NWfb=unscale(s, std_errors_individual_vars_NWfb, H, demean, scale);
  arma::mat intervals_NWfb(PDL.h, 2*1+1); // intervals for individual variables
  intervals_NWfb.col(1)=PDL.bhat_1;
  for(unsigned int p=0; p<PDL.h; p++){
    for(unsigned int j=0; j<1; j++){
      intervals_NWfb(p,j)=PDL.bhat_1(p)-cv*std_errors_individual_vars_NWfb(p);
      intervals_NWfb(p, 2*1-j)=PDL.bhat_1(p)+cv*std_errors_individual_vars_NWfb(p);
    }
  }
  arma::mat intervals_unscaled_NWfb=intervals_NWfb; //unscaling works with just the normal unscale function
  for(int j=0; j<2*1+1; j++){
    intervals_unscaled_NWfb.col(j)=unscale(s, intervals_NWfb.col(j), H, demean, scale);
  }
  //////////////////////////////////////////////////////

  partial_desparsified_lasso_inference_output ret;
  ret.init_partial=PDL.init_partial;
  //ret.nw_partials=PDL.nw_partials;
  ret.N=PDL.N;
  ret.T_=PDL.T_;
  ret.h=PDL.h;
  ret.init_gridsize=PDL.init_gridsize;
  ret.init_lambda_pos=PDL.init_lambda_pos;
  ret.init_nonzero=PDL.init_nonzero;
  ret.init_lambda=PDL.init_lambda;
  ret.init_nonzero_limit=PDL.init_nonzero_limit;
  ret.init_criterion_value=PDL.init_criterion_value;
  ret.init_SSR=PDL.init_SSR;
  ret.bhat_1=PDL.bhat_1;
  ret.bhat_1_unscaled=bhat_1_unscaled;
  ret.init_grid=PDL.init_grid;
  ret.y=PDL.y;
  ret.init_residual=PDL.init_residual;
  ret.betahat=PDL.betahat;
  ret.betahat_1=PDL.betahat_1;
  ret.betahat_2=PDL.betahat_2;
  ret.nw_gridsizes=PDL.nw_gridsizes;
  ret.nw_lambda_poss=PDL.nw_lambda_poss;
  ret.nw_nonzeros=PDL.nw_nonzeros;
  ret.nw_lambdas=PDL.nw_lambdas;
  ret.nw_nonzero_limits=PDL.nw_nonzero_limits;
  ret.nw_criterion_values=PDL.nw_criterion_values;
  ret.nw_SSRs=PDL.nw_SSRs;
  ret.H=PDL.H;
  ret.minusH=PDL.minusH;
  ret.X=PDL.X;
  ret.X_1=PDL.X_1;
  ret.X_2=PDL.X_2;
  ret.gammahats=PDL.gammahats;
  ret.Gammahat=PDL.Gammahat;
  ret.Upsilonhat_inv=PDL.Upsilonhat_inv;
  ret.Thetahat=PDL.Thetahat;
  ret.nw_grids=PDL.nw_grids;
  ret.nw_residuals=PDL.nw_residuals;
  ret.init_opt_type=PDL.init_opt_type;
  ret.init_selection_type=PDL.init_selection_type;
  ret.nw_opt_types=PDL.nw_opt_types;
  ret.nw_selection_types=PDL.nw_selection_types;

  ret.Omegahat=Omegahat;
  ret.R=R;
  ret.q=q;
  ret.std_errors_individual_vars=std_errors_individual_vars_unscale;
  ret.z_quantiles=z_quantiles;
  ret.intervals=intervals;
  ret.intervals_unscaled=intervals_unscaled;
  ret.intervals_Rq=intervals_Rq;
  ret.intervals_Rq_unscaled=intervals_Rq_unscaled;
  ret.z_stats_Rq=z_stats_Rq;
  ret.joint_chi2_stat=joint_chi2_stat;
  ret.chi2_quantiles=chi2_quantiles;
  ret.init_nonzero_pos=PDL.init_nonzero_pos;
  ret.nw_nonzero_poss=PDL.nw_nonzero_poss;

  ret.intervals_EWC=intervals_EWC;
  ret.intervals_unscaled_EWC=intervals_unscaled_EWC;
  ret.Omegahat_EWC=Omegahat_EWC;

  ret.intervals_NWfb=intervals_NWfb;
  ret.intervals_unscaled_NWfb=intervals_unscaled_NWfb;
  ret.Omegahat_NWfb=Omegahat_NWfb;
  return ret;
}

arma::mat na_matrix(unsigned int rows, unsigned int cols){
  NumericMatrix m(rows,cols) ;
  std::fill( m.begin(), m.end(), NumericVector::get_na() ) ;
  arma::mat ret=as<arma::mat>(m);
  return ret ;
}

arma::mat naomit(arma::mat x){
  unsigned int T_=x.n_rows;
  arma::uvec na_indexes;
  arma::uvec index(1);
  for(unsigned int i=0; i<T_; i++){
    if((x.row(i)).has_nan()){
      index(0)=i;
      na_indexes=arma::join_vert(na_indexes,index);
    }
  }
  x.shed_rows(na_indexes);
  return x;
}

arma::mat Rcpp_make_lags(const arma::mat& x, const unsigned int& lags) {
  unsigned int N=x.n_cols;
  unsigned int T_=x.n_rows;
  //arma::mat ret=na_matrix(T_, N*lags);
  arma::mat ret=mat(T_, N*lags, fill::zeros);
  for(unsigned int i=1; i<=lags; i++){
    //ret[(1+i):T_,((i-1)*N+1):(i*N)]<-x[1:(T_-i),]
    ret(span(i,T_-1),span((i-1)*N,i*N-1))=x(span(0,(T_-1-i)),span(0,N-1));
  }
  return ret;
}

struct LP_output{
  arma::mat intervals;
  NumericMatrix manual_Thetahat;
  List betahats;
};



arma::mat dummify(const arma::mat& dummy, const arma::mat& M){
  arma::mat ret(M.n_rows, dummy.n_cols*M.n_cols);
  for(unsigned int i=0; i<M.n_cols; i++){
    for(unsigned int j=0; j<dummy.n_cols; j++){
      ret.col(i*dummy.n_cols+j)=dummy.col(j)%M.col(i);
    }
  }
  return ret;
}

struct LP_state_dependent_output{
  arma::cube intervals;
  arma::cube intervals_EWC;
  arma::cube intervals_NWfb;
  arma::mat manual_Thetahat;
  arma::mat betahats;
  arma::mat regressors;
  arma::mat regressors_trimmed;
  arma::mat init_residuals;
  arma::mat nw_residuals;
  arma::mat dependents;
};

reg_output OLS_HAC(const arma::mat& X, const arma::colvec& y, const arma::uvec& H, const bool& demean, const bool& scale,
                   const double& LRVtrunc, const double& T_multiplier, const NumericVector alphas){
  standardize_output s=standardize(X, y, demean, scale);
  unsigned int uiT=X.n_rows;
  unsigned int uiN=X.n_cols;
  double T_=static_cast<double>(uiT);
  double N_=static_cast<double>(uiN);
  arma::mat X_=s.X_scaled;
  arma::vec y_=s.y_scaled;
  arma::mat Sigmahat=X_.t()*X_/T_;
  arma::mat Thetahat=inv(Sigmahat);
  arma::vec betahat=Thetahat*(X_.t()*y_/T_);
  arma::vec uhat=y_-X_*betahat;
  arma::mat Omegahat=LRVestimator(uhat, X_, N_, T_, N_, LRVtrunc, T_multiplier);
  arma::mat V=Thetahat*Omegahat*Thetahat.t();
  arma::vec std_errors(H.n_elem);
  for(unsigned int i=0; i<H.n_elem; i++){
    std_errors(i)=sqrt(V(i,i)/T_);
  }
  arma::vec b_H=betahat.rows(H);
  arma::vec z_quantiles=qnorm(alphas/2.0,0.0,1.0,false,false);
  arma::mat intervals(H.n_elem,2*alphas.length()+1);
  intervals.col(alphas.length())=b_H;
  for(unsigned int i=0; i<H.n_elem; i++){
    for(int j=0; j<alphas.length(); j++){
      intervals(i,j)=b_H(i)-z_quantiles(j)*std_errors(i);
      intervals(i, 2*alphas.length()-j)=b_H(i)+z_quantiles(j)*std_errors(i);
    }
  }
  arma::mat intervals_unscaled=intervals;
  for(int j=0; j<2*alphas.length()+1; j++){
    intervals_unscaled.col(j)=unscale(s, intervals.col(j), H, demean, scale);
  }
  arma::vec b_H_unscaled=unscale(s, b_H, H, demean, scale);
  //List init=List::create(Named("betahat")=betahat
  //);
  //List inference=List::create(Named("Omegahat")=Omegahat,
  //                            Named("z_quantiles")=z_quantiles,
  //                            Named("intervals")=intervals,
  //                            Named("intervals_unscaled")=intervals_unscaled
  //);
  //return List::create(Named("inference")=inference,
  //                    Named("init")=init,
  //                   Named("b_H")=b_H,
  //                    Named("b_H_unscaled")=b_H_unscaled,
  //                    Named("Thetahat")=Thetahat
  //);

  ////this part deals with doing inference via the EWC instead of NW
  arma::mat Omegahat_EWC=LRVestimatorEWC(uhat, X_, N_, T_, N_, 2/3, 0.4);
  double nu=std::ceil(0.4*pow(double(T_),2/3));
  arma::vec t_nu_quantiles=qt(alphas/2.0,nu,false,false);
  arma::mat covariance_EWC=Thetahat*Omegahat_EWC*Thetahat;
  arma::vec std_errors_individual_vars_EWC(H.n_elem);
  for(unsigned int i=0; i<H.n_elem; i++){
    std_errors_individual_vars_EWC(i)=sqrt( as_scalar(covariance_EWC(i,i))/double(T_) );
  }
  arma::vec std_errors_individual_vars_unscale_EWC=unscale(s, std_errors_individual_vars_EWC, H, demean, scale);
  arma::mat intervals_EWC(H.n_elem, 2*alphas.length()+1); // intervals for individual variables
  intervals_EWC.col(alphas.length())=b_H;
  for(unsigned int p=0; p<H.n_elem; p++){
    for(unsigned int j=0; j<alphas.length(); j++){
      intervals_EWC(p,j)=b_H(p)-t_nu_quantiles(j)*std_errors_individual_vars_EWC(p);
      intervals_EWC(p, 2*alphas.length()-j)=b_H(p)+t_nu_quantiles(j)*std_errors_individual_vars_EWC(p);
    }
  }
  arma::mat intervals_unscaled_EWC=intervals_EWC; //unscaling works with just the normal unscale function
  for(int j=0; j<2*alphas.length()+1; j++){
    intervals_unscaled_EWC.col(j)=unscale(s, intervals_EWC.col(j), H, demean, scale);
  }
  ////////////////////////////////////////////////////////////////////


  ////this part deals with the fixed-b asymptotics of NW
  arma::mat Omegahat_NWfb=LRVestimator(uhat, X_, N_, T_, N_, 1/2, 1.3);
  double S=std::ceil(1.3*pow(double(T_),1/2));
  double a0=1.96, a1=2.9694, a2=0.3142, a3=-0.3427; //taken from Table 1 of Kiefer and Vogelsang (2005)
  double b=S/T_;
  double cv=a0+a1*b+a2*pow(b,2)+a3*pow(b,3);
  arma::mat covariance_NWfb=Thetahat*Omegahat_NWfb*Thetahat;
  arma::vec std_errors_individual_vars_NWfb(H.n_elem);
  for(unsigned int i=0; i<H.n_elem; i++){
    std_errors_individual_vars_NWfb(i)=sqrt( as_scalar(covariance_NWfb(i,i))/double(T_) );
  }
  arma::vec std_errors_individual_vars_unscale_NWfb=unscale(s, std_errors_individual_vars_NWfb, H, demean, scale);
  arma::mat intervals_NWfb(H.n_elem, 2*1+1); // intervals for individual variables
  intervals_NWfb.col(1)=b_H;
  for(unsigned int p=0; p<H.n_elem; p++){
    for(unsigned int j=0; j<1; j++){
      intervals_NWfb(p,j)=b_H(p)-cv*std_errors_individual_vars_NWfb(p);
      intervals_NWfb(p, 2*1-j)=b_H(p)+cv*std_errors_individual_vars_NWfb(p);
    }
  }
  arma::mat intervals_unscaled_NWfb=intervals_NWfb; //unscaling works with just the normal unscale function
  for(int j=0; j<2*1+1; j++){
    intervals_unscaled_NWfb.col(j)=unscale(s, intervals_NWfb.col(j), H, demean, scale);
  }
  //////////////////////////////////////////////////////

  reg_output o;
  o.Omegahat=Omegahat;
  o.z_quantiles=z_quantiles;
  o.intervals=intervals;
  o.intervals_unscaled=intervals_unscaled;
  o.intervals_unscaled_EWC=intervals_unscaled_EWC;
  o.intervals_unscaled_NWfb=intervals_unscaled_NWfb;
  o.betahat=betahat;
  o.b_H=b_H;
  o.b_H_unscaled=b_H_unscaled;
  o.Thetahat=Thetahat;
  return o;
}

LP_state_dependent_output local_projection_state_dependent(Nullable<NumericMatrix> r_, const arma::vec& x, const arma::vec& y, Nullable<NumericMatrix> q_, Nullable<NumericMatrix> state_dummy_,
                                                           const bool& y_predetermined,const bool& cumulate_y, const unsigned int& hmax,
                                                           const unsigned int& lags,const NumericVector& alphas, const bool& init_partial, const int& selection, const double& PIconstant,
                                                           const bool& progress_bar, bool OLS, unsigned int threads, arma::mat seeds_mat){
  arma::mat w;
  arma::mat w_lags;
  arma::mat regressors;
  arma::vec dependent;
  arma::mat joint_trimmed;
  arma::mat dependent_trimmed;
  arma::mat regressors_trimmed;
  arma::mat tempmat;
  arma::mat mThetahat;
  arma::mat mUpsilonhat_inv;
  arma::mat mnw_residuals;
  arma::uvec H;
  arma::mat state_dummy;
  unsigned int states=1;
  if(state_dummy_.isNotNull()){
    NumericMatrix state_dummy_temp(state_dummy_);
    state_dummy=as<arma::mat>(state_dummy_temp);
    states=state_dummy.n_cols;
    H=linspace<arma::uvec>(0,2*state_dummy.n_cols-2,2*state_dummy.n_cols-1);
  }else{
    H=linspace<arma::uvec>(0,0,1);
  }
  arma::mat r;
  if(r_.isNotNull()){
    NumericMatrix r_temp(r_);
    r=as<arma::mat>(r_temp);
  }
  arma::mat q;
  if(q_.isNotNull()){
    NumericMatrix q_temp(q_);
    q=as<arma::mat>(q_temp);
  }

  LogicalVector nw_partials(H.n_elem);
  for(unsigned int i=0; i<H.n_elem; i++){
    nw_partials(i)=false;
  }
  int init_selection_type=selection;
  arma::vec nw_selection_types(H.n_elem);
  for(unsigned int i=0; i<H.n_elem; i++){
    nw_selection_types(i)=false;
  }
  double init_nonzero_limit=0.5;
  arma::vec nw_nonzero_limits(H.n_elem);
  for(unsigned int i=0; i<H.n_elem; i++){
    nw_nonzero_limits(i)=0.5;
  }
  double init_opt_threshold=1e-4;
  arma::vec nw_opt_thresholds(H.n_elem);
  for(unsigned int i=0; i<H.n_elem; i++){
    nw_opt_thresholds(i)=1e-4;
  }
  int init_opt_type=3;
  arma::vec nw_opt_types(H.n_elem);
  for(unsigned int i=0; i<H.n_elem; i++){
    nw_opt_types(i)=3;
  }
  arma::mat intervals_zeros(hmax+1,1+2*as<arma::mat>(alphas).n_elem, fill::zeros);
  arma::mat intervals_zeros_NWfb(hmax+1,1+2*1, fill::zeros);//////////////////////////
  arma::cube intervals(hmax+1,1+2*as<arma::mat>(alphas).n_elem, states);
  arma::cube intervals_EWC(hmax+1,1+2*as<arma::mat>(alphas).n_elem, states);////////////////////////////////
  arma::cube intervals_NWfb(hmax+1,1+2*1, states);////////////////////////////////
  for(unsigned int i=0; i<states; i++){
    intervals.slice(i)=intervals_zeros;
    intervals_EWC.slice(i)=intervals_zeros;//////////////////////////////
    intervals_NWfb.slice(i)=intervals_zeros_NWfb;//////////////////////////////
  }
  arma::mat R(H.n_elem, H.n_elem, fill::eye);
  arma::vec Q(H.n_elem, fill::ones);
  unsigned int T_=y.n_elem;
  //check if any columns in x y are the same
  bool is_same=false;
  if(sum(abs(x-y))<1e-8){
    is_same=true;
  }

  //Set up the regressors
  if(is_same){ //x=y, y is not predetermined w.r.t. x or vice versa

    w=join_horiz(r,x,q); //skip the y, because it would be the same variable twice
    w_lags=Rcpp_make_lags(w,lags);
    if(state_dummy_.isNotNull()){ //if there are states
      regressors=join_horiz(dummify(state_dummy, x),
                            state_dummy.cols(0, state_dummy.n_cols-2),
                            dummify(state_dummy, join_horiz(r, w_lags))); //y is NOT included contemporaneously, because x=y already is
    }else{//if there are no states
      regressors=join_horiz(x, r, w_lags);
    }
  }else{ //x and y are different

    w=join_horiz(r,x,y,q); //include both x and y
    w_lags=Rcpp_make_lags(w,lags);
    if(y_predetermined){ //y is predetermined wrt x.  //y IS included contemporaneously. since it is assumed to be predetermined wrt x, there are no endogeneity issues with including it
      if(state_dummy_.isNotNull()){ //if there are states
        regressors=join_horiz(dummify(state_dummy, x),
                              state_dummy.cols(0, state_dummy.n_cols-2),
                              dummify(state_dummy, join_horiz(r, w_lags,y)));
      }else{//if there are no states
        regressors=join_horiz(x, r, w_lags, y);
      }
    }else{// x is predetermined wrt y, this is the standard setup in eq. 1 of PM&W
      if(state_dummy_.isNotNull()){ //if there are states
        regressors=join_horiz(dummify(state_dummy, x),
                              state_dummy.cols(0, state_dummy.n_cols-2),
                              dummify(state_dummy, join_horiz(r, w_lags))); //y is NOT included contemporaneously, because there would be issues with endogeneity
      }else{//if there are no states
        regressors=join_horiz(x, r, w_lags);
      }
    }
  }

  //Setting up the dependent for the initial estimation

  //We actually estimate the impulse responses at horizon 0 if we're in the standard PM&W setup: y_predetermined=FALSE, and y!=x
  //We also estimate at horizon 0 if y_predetermined=FALSE and we have multiple states (even when y=x)
  if(!y_predetermined && (!is_same || states>1)){
    dependent=y; //cumulating doesn't matter at horizon 0
    joint_trimmed=join_horiz(dependent,regressors); joint_trimmed=joint_trimmed.rows(lags,T_-0-1);
  }else{ //in other cases, we want to run a dummy regression at horizon 1, just to get the nodewise parts
    dependent=mat(T_,1,fill::zeros);
    if(cumulate_y){ //cumulation matters at horizon 1
      for(unsigned int i=0; i<T_-1;i++){
        dependent.row(i)=sum(y.rows(i,i+1));
      }
    }else{
      for(unsigned int i=0; i<T_-1;i++){
        dependent.row(i)=y.row(i+1);
      }
    }
    joint_trimmed=join_horiz(dependent,regressors); joint_trimmed=joint_trimmed.rows(lags,T_-1-1);
  }
  dependent_trimmed=joint_trimmed.col(0);
  regressors_trimmed=joint_trimmed.cols(1, joint_trimmed.n_cols-1);



  // run the actual estimation
  reg_output d;
  if(OLS && regressors_trimmed.n_cols>regressors_trimmed.n_rows){ //if the sample size is too small, don't do OLS. This also switches OLS off for all later steps
    //warning("number of variables larger than sample size, taking OLS=FALSE to prevent errors");
    OLS=false;
  }
  if(OLS){
    d=OLS_HAC(regressors_trimmed, dependent_trimmed, H, true, true, 0, 0, alphas);
  }else{
    nodewise_manual nm;
    nm.use_manual=false;
    grids_output g=build_gridsXy(regressors_trimmed.n_rows, regressors_trimmed.n_cols, 50, regressors_trimmed, dependent_trimmed, H, true, true);
    partial_desparsified_lasso_inference_output pdli=partial_desparsified_lasso_inference(regressors_trimmed, dependent_trimmed, H, true, true, init_partial, nw_partials,
                                                                                          g.init_grid, g.nw_grids, init_selection_type, nw_selection_types,
                                                                                          init_nonzero_limit, nw_nonzero_limits, init_opt_threshold, nw_opt_thresholds, init_opt_type, nw_opt_types,
                                                                                          0, 0, alphas, R, Q, PIconstant, 0.05, progress_bar, threads, seeds_mat.col(0),
                                                                                          nm);
    mThetahat=pdli.Thetahat;
    mUpsilonhat_inv=pdli.Upsilonhat_inv;
    mnw_residuals=pdli.nw_residuals;
    d.Omegahat=pdli.Omegahat;
    d.z_quantiles=pdli.z_quantiles;
    d.intervals=pdli.intervals;
    d.intervals_unscaled=pdli.intervals_unscaled;
    d.intervals_unscaled_EWC=pdli.intervals_unscaled_EWC;
    d.intervals_unscaled_NWfb=pdli.intervals_unscaled_NWfb;
    d.betahat=pdli.betahat;
    d.b_H=pdli.bhat_1;
    d.b_H_unscaled=pdli.bhat_1_unscaled;
    d.Thetahat=pdli.Thetahat;
  }
  arma::mat betahats(regressors_trimmed.n_cols,hmax+1,fill::zeros);
  betahats.col(0)=d.betahat;

  //fill in the results if they're needed
  if(!y_predetermined && (!is_same || states>1)){
      for(unsigned int i=0; i<states; i++){
        intervals(span(0,0),span(0,1+2*as<arma::mat>(alphas).n_elem-1),span(i,i))=(d.intervals_unscaled).row(i);
        intervals_EWC(span(0,0),span(0,1+2*as<arma::mat>(alphas).n_elem-1),span(i,i))=(d.intervals_unscaled_EWC).row(i);
        intervals_NWfb(span(0,0),span(0,1+2*1-1),span(i,i))=(d.intervals_unscaled_NWfb).row(i);
    }
  }

  //filling in either 1's or 0's in other cases
  if(!y_predetermined && is_same && states==1){ //fill with 1's
    arma::mat mat_of_ones(1,1+2*as<arma::mat>(alphas).n_elem, fill::ones);
    arma::mat mat_of_ones_NWfb(1,1+2*1, fill::ones);
    for(unsigned int i=0; i<states; i++){ //note that this loop is only over 1 element
      intervals(span(0,0),span(0,1+2*as<arma::mat>(alphas).n_elem-1),span(i,i))=mat_of_ones.row(0);
      intervals_EWC(span(0,0),span(0,1+2*as<arma::mat>(alphas).n_elem-1),span(i,i))=mat_of_ones.row(0);//////////////////
      intervals_NWfb(span(0,0),span(0,1+2*1-1),span(i,i))=mat_of_ones_NWfb.row(0);//////////////////
    }
  }

  if(y_predetermined){
    arma::mat mat_of_ones(1,1+2*as<arma::mat>(alphas).n_elem, fill::ones);
    arma::mat mat_of_ones_NWfb(1,1+2*1, fill::ones);
    if(is_same){//fill with 1's
      for(unsigned int i=0; i<states; i++){
        intervals(span(0,0),span(0,1+2*as<arma::mat>(alphas).n_elem-1),span(i,i))=(d.intervals_unscaled).row(i);
        intervals_EWC(span(0,0),span(0,1+2*as<arma::mat>(alphas).n_elem-1),span(i,i))=(d.intervals_unscaled_EWC).row(i);
        intervals_NWfb(span(0,0),span(0,1+2*1-1),span(i,i))=(d.intervals_unscaled_NWfb).row(i);
      }
    }else{ //fill with 0's
      //do nothing, because the matrices already start out filled with 0's
    }
  }

  //now estimate the nodewise regressions at different horizons
  Progress p(hmax, progress_bar);
  if(threads>0){
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
# pragma omp parallel for schedule(dynamic)
    for(unsigned int h=1; h<=hmax;h++){
      if (!Progress::check_abort()){
        reg_output d_p;
        arma::mat dependent_p;
        dependent_p=mat(T_,1,fill::zeros);
        if(cumulate_y){
          for(unsigned int i=0; i<T_-h;i++){
            dependent_p.row(i)=sum(y.rows(i,i+h));
          }
        }else{
          for(unsigned int k=0; k<T_-h;k++){
            dependent_p.row(k)=y.row(k+h);
          }
        }
        arma::mat joint_trimmed_p=join_horiz(dependent_p,regressors); joint_trimmed_p=joint_trimmed_p.rows(lags,T_-h-1);
        arma::mat dependent_trimmed_p=joint_trimmed_p.col(0);
        arma::mat regressors_trimmed_p=joint_trimmed_p.cols(1, joint_trimmed.n_cols-1);
        if(OLS){
          d_p=OLS_HAC(regressors_trimmed_p, dependent_trimmed_p, H, true, true, 0, 0, alphas);
        }else{
          nodewise_manual nm_p;
          nm_p.use_manual=true;
          nm_p.manual_Thetahat=mThetahat;
          nm_p.manual_Upsilonhat_inv=mUpsilonhat_inv;
          nm_p.manual_nw_residuals=mnw_residuals(span(0,regressors_trimmed_p.n_rows-1), span(0,H.n_elem-1));
          grids_output g_p=build_gridsXy(regressors_trimmed_p.n_rows, regressors_trimmed_p.n_cols, 50, regressors_trimmed_p, dependent_trimmed_p, H, true, true);
          partial_desparsified_lasso_inference_output pdli_p=partial_desparsified_lasso_inference(regressors_trimmed_p, dependent_trimmed_p, H, true, true, init_partial, nw_partials,
                                                                                                  g_p.init_grid, g_p.nw_grids, init_selection_type, nw_selection_types,
                                                                                                  init_nonzero_limit, nw_nonzero_limits, init_opt_threshold, nw_opt_thresholds, init_opt_type, nw_opt_types,
                                                                                                  0, 0, alphas, R, Q, PIconstant, 0.05, false, 0, seeds_mat.col(h),
                                                                                                  nm_p);
          d_p.Omegahat=pdli_p.Omegahat;
          d_p.z_quantiles=pdli_p.z_quantiles;
          d_p.intervals=pdli_p.intervals;
          d_p.intervals_unscaled=pdli_p.intervals_unscaled;
          d_p.intervals_unscaled_EWC=pdli_p.intervals_unscaled_EWC;/////////////////
          d_p.intervals_unscaled_NWfb=pdli_p.intervals_unscaled_NWfb;////////////////
          d_p.betahat=pdli_p.betahat;
          d_p.b_H=pdli_p.bhat_1;
          d_p.b_H_unscaled=pdli_p.bhat_1_unscaled;
          d_p.Thetahat=pdli_p.Thetahat;
        }
        betahats.col(h)=d_p.betahat;
        for(unsigned int j=0; j<states; j++){
          intervals(span(h,h),span(0,1+2*as<arma::mat>(alphas).n_elem-1),span(j,j))=(d_p.intervals_unscaled).row(j);
          intervals_EWC(span(h,h),span(0,1+2*as<arma::mat>(alphas).n_elem-1),span(j,j))=(d_p.intervals_unscaled_EWC).row(j);//////////
          intervals_NWfb(span(h,h),span(0,1+2*1-1),span(j,j))=(d_p.intervals_unscaled_NWfb).row(j);/////////
        }
        p.increment();
      }
    }
  }else{
    for(unsigned int h=1; h<=hmax;h++){
      reg_output d_p;
      arma::mat dependent_p;
      dependent_p=mat(T_,1,fill::zeros);
      if(cumulate_y){
        for(unsigned int i=0; i<T_-h;i++){
          dependent_p.row(i)=sum(y.rows(i,i+h));
        }
      }else{
        for(unsigned int k=0; k<T_-h;k++){
          dependent_p.row(k)=y.row(k+h);
        }
      }
      arma::mat joint_trimmed_p=join_horiz(dependent_p,regressors); joint_trimmed_p=joint_trimmed_p.rows(lags,T_-h-1);
      arma::mat dependent_trimmed_p=joint_trimmed_p.col(0);
      arma::mat regressors_trimmed_p=joint_trimmed_p.cols(1, joint_trimmed.n_cols-1);
      if(OLS){
        d_p=OLS_HAC(regressors_trimmed_p, dependent_trimmed_p, H, true, true, 0, 0, alphas);
      }else{
        nodewise_manual nm_p;
        nm_p.use_manual=true;
        nm_p.manual_Thetahat=mThetahat;
        nm_p.manual_Upsilonhat_inv=mUpsilonhat_inv;
        nm_p.manual_nw_residuals=mnw_residuals(span(0,regressors_trimmed_p.n_rows-1), span(0,H.n_elem-1));
        grids_output g_p=build_gridsXy(regressors_trimmed_p.n_rows, regressors_trimmed_p.n_cols, 50, regressors_trimmed_p, dependent_trimmed_p, H, true, true);
        partial_desparsified_lasso_inference_output pdli_p=partial_desparsified_lasso_inference(regressors_trimmed_p, dependent_trimmed_p, H, true, true, init_partial, nw_partials,
                                                                                                g_p.init_grid, g_p.nw_grids, init_selection_type, nw_selection_types,
                                                                                                init_nonzero_limit, nw_nonzero_limits, init_opt_threshold, nw_opt_thresholds, init_opt_type, nw_opt_types,
                                                                                                0, 0, alphas, R, Q, PIconstant, 0.05, false, 0, seeds_mat.col(h),
                                                                                                nm_p);
        d_p.Omegahat=pdli_p.Omegahat;
        d_p.z_quantiles=pdli_p.z_quantiles;
        d_p.intervals=pdli_p.intervals;
        d_p.intervals_unscaled=pdli_p.intervals_unscaled;
        d_p.intervals_unscaled_EWC=pdli_p.intervals_unscaled_EWC;//////////////
        d_p.intervals_unscaled_NWfb=pdli_p.intervals_unscaled_NWfb;//////////////
        d_p.betahat=pdli_p.betahat;
        d_p.b_H=pdli_p.bhat_1;
        d_p.b_H_unscaled=pdli_p.bhat_1_unscaled;
        d_p.Thetahat=pdli_p.Thetahat;
      }
      betahats.col(h)=d_p.betahat;
      for(unsigned int j=0; j<states; j++){
        intervals(span(h,h),span(0,1+2*as<arma::mat>(alphas).n_elem-1),span(j,j))=(d_p.intervals_unscaled).row(j);
        intervals_EWC(span(h,h),span(0,1+2*as<arma::mat>(alphas).n_elem-1),span(j,j))=(d_p.intervals_unscaled_EWC).row(j);//////////
        intervals_NWfb(span(h,h),span(0,1+2*1-1),span(j,j))=(d_p.intervals_unscaled_NWfb).row(j);/////////
      }
      p.increment();
    }
  }
  LP_state_dependent_output LPsdo;
  LPsdo.intervals=intervals;
  LPsdo.intervals_EWC=intervals_EWC;///////////////
  LPsdo.intervals_NWfb=intervals_NWfb;//////////////
  LPsdo.manual_Thetahat=mThetahat;
  LPsdo.betahats=betahats;
  LPsdo.regressors=regressors;
  LPsdo.regressors_trimmed=regressors_trimmed;
  //LPsdo.init_residuals=init_residuals;
  //LPsdo.nw_residuals=nw_residuals;
  //LPsdo.dependents=dependents;
  return LPsdo;
}


//wrapper functions

// //' Calculate the desparsified lasso
// //' @param X regressor matrix
// //' @param y dependent variable vector
// //' @param H indexes of relevant regressors
// //' @param init_partial (optional) boolean, true if you want the initial lasso to be partially penalized (false by default)
// //' @param nw_partials (optional) boolean vector with the dimension of H, trues if you want the nodewise regressions to be partially penalized (all false by default)
// //' @param gridsize (optional) integer, how many different lambdas there should be in both initial and nodewise grids (100 by default)
// //' @param init_grid (optional) vector, containing user specified initial grid
// //' @param nw_grids (optional) matrix with number of rows the size of H, rows containing user specified grids for the nodewise regressions
// //' @param init_selection_type (optional) integer, how should lambda be selected in the initial regression, 1=BIC, 2=AIC, 3=EBIC (1 by default)
// //' @param nw_selection_types (optional) integer vector with the dimension of H, how should lambda be selected in the nodewise regressions, 1=BIC, 2=AIC, 3=EBIC (all 1s by default)
// //' @param init_nonzero_limit (optional) number controlling the maximum number of nonzeros that can be selected in the initial regression (0.5 by default, meaning no more than 0.5*T_ regressors can have nonzero estimates)
// //' @param nw_nonzero_limits (optional) vector with the dimension of H, controlling the maximum number of nonzeros that can be selected in the nodewise regressions (0.5s by default)
// //' @param init_opt_threshold (optional) optimization threshold for the coordinate descent algorithm in the initial regression (10^(-4) by default)
// //' @param nw_opt_thresholds (optional) vector with the dimension of H, optimization thresholds for the coordinate descent algorithm in the nodewise lasso regression (10^(-4)s by default)
// //' @param init_opt_type (optional) integer, which type of coordinate descent algorithm should be used in the initial regression, 1=naive, 2=covariance, 3=adaptive (3 by default)
// //' @param nw_opt_types (optional)integer vector with the dimension of H, which type of coordinate descent algorithm should be used in the nodewise regressions, 1=naive, 2=covariance, 3=adaptive (3s by default)
// //' @param LRVtrunc (optional) parameter controlling the bandwidth Q_T used in the long run covariance matrix, Q_T=ceil(T_multiplier*T_^LRVtrunc) (LRVtrunc=0.2 by default)
// //' @param T_multiplier (optional) parameter controlling the bandwidth Q_T used in the long run covariance matrix, Q_T=ceil(T_multiplier*T_^LRVtrunc) (T_multiplier=2 by default)
// //' @param alphas (optional) vector of significance levels (c(0.01,0.05,0.1) by default)
// //' @param R (optional) matrix with number of columns the dimension of H, used to test the null hypothesis R*beta=q (identity matrix as default)
// //' @param q (optional) vector of size same as the rows of H, used to test the null hypothesis R*beta=q (zeroes by default)
// //' @param manual_Thetahat_ (optional) matrix with rows the size of H and columns the number of regressors. Can be obtained from earlier executions of the function to avoid unnecessary calculations of the nodewise regressions (NULL as default)
// //' @param manual_Upsilonhat_inv_ (optional) matrix with rows and columns the size of H. Can be obtained from earlier executions of the function to avoid unnecessary calculations of the nodewise regressions (NULL as default)
// //' @param manual_nw_residuals (optional) matrix with rows equal to the sample size and columns the size of H, containing the residuals from the nodewise regressions. Can be obtained from earlier executions of the function to avoid unnecessary calculations of the nodewise regressions (NULL as default)
// [[Rcpp::export(.Rwrap_partial_desparsified_lasso_inference)]]
List Rwrap_partial_desparsified_lasso_inference(const arma::mat& X, const arma::colvec& y, const arma::uvec& H, const bool& demean, const bool& scale, const bool& init_partial, const LogicalVector& nw_partials,
                                                const arma::vec& init_grid, const arma::mat& nw_grids, const int& init_selection_type, const arma::vec& nw_selection_types,
                                                const double& init_nonzero_limit, const arma::vec& nw_nonzero_limits, const double& init_opt_threshold, const arma::vec& nw_opt_thresholds, const int& init_opt_type, const arma::vec& nw_opt_types,
                                                const double& LRVtrunc, const double& T_multiplier, const NumericVector alphas, const arma::mat& R, const arma::vec& q, const double& PIconstant, const double& PIprobability, bool progress_bar, unsigned int threads, arma::vec seeds,
                                                Nullable<NumericMatrix> manual_Thetahat_, Nullable<NumericMatrix> manual_Upsilonhat_inv_, Nullable<NumericMatrix> manual_nw_residuals_){
  //If we manually provide ALL the necessary nodewise components, we don't need to estimate them any more
  nodewise_manual nm;
  if(manual_Thetahat_.isNotNull() && manual_Upsilonhat_inv_.isNotNull() && manual_nw_residuals_.isNotNull()){
    nm.use_manual=true;
    NumericMatrix temp_manual_Thetahat(manual_Thetahat_);  // conversion from Nullable<NumericMatrix> to NumericMatrix
    nm.manual_Thetahat=as<arma::mat>(temp_manual_Thetahat); //conversion from NumericMatrix to arma::mat

    NumericMatrix temp_manual_Upsilonhat_inv(manual_Upsilonhat_inv_);   // conversion from Nullable<NumericMatrix> to NumericMatrix
    nm.manual_Upsilonhat_inv=as<arma::mat>(temp_manual_Upsilonhat_inv); //conversion from NumericMatrix to arma::mat

    NumericMatrix temp_manual_nw_residuals(manual_nw_residuals_); // conversion from Nullable<NumericMatrix> to NumericMatrix
    nm.manual_nw_residuals=as<arma::mat>(temp_manual_nw_residuals); //conversion from NumericMatrix to arma::
  }else{
    nm.use_manual=false;
  }
  partial_desparsified_lasso_inference_output PDLI=partial_desparsified_lasso_inference(X, y, H, demean, scale, init_partial, nw_partials,
                                                                                        init_grid, nw_grids, init_selection_type, nw_selection_types,
                                                                                        init_nonzero_limit, nw_nonzero_limits, init_opt_threshold, nw_opt_thresholds, init_opt_type, nw_opt_types,
                                                                                        LRVtrunc, T_multiplier, alphas, R, q, PIconstant, PIprobability, progress_bar, threads, seeds,
                                                                                        //manual_Thetahat_, manual_Upsilonhat_inv_, manual_nw_residuals_
                                                                                        nm);
  List misc=List::create(Named("N")=PDLI.N,
                         Named("T_")=PDLI.T_,
                         Named("h")=PDLI.h,
                         Named("y")=PDLI.y,
                         Named("H")=PDLI.H,
                         Named("minusH")=PDLI.minusH,
                         Named("X")=PDLI.X,
                         Named("X_1")=PDLI.X_1,
                         Named("X_2")=PDLI.X_2
  );
  List init=List::create(Named("partial")=PDLI.init_partial,
                         Named("gridsize")=PDLI.init_gridsize,
                         Named("lambda_pos")=PDLI.init_lambda_pos,
                         Named("nonzero")=PDLI.init_nonzero,
                         Named("lambda")=PDLI.init_lambda,
                         Named("nonzero_limit")=PDLI.init_nonzero_limit,
                         Named("criterion_value")=PDLI.init_criterion_value,
                         Named("SSR")=PDLI.init_SSR,
                         Named("grid")=PDLI.init_grid,
                         Named("residual")=PDLI.init_residual,
                         Named("betahat")=PDLI.betahat,
                         Named("betahat_1")=PDLI.betahat_1,
                         Named("betahat_2")=PDLI.betahat_2,
                         Named("opt_type")=PDLI.init_opt_type,
                         Named("selection_type")=PDLI.init_selection_type,
                         Named("nonzero_pos")=PDLI.init_nonzero_pos
  );
  List nw=List::create(//Named("partials")=PDLI.nw_partials,
    Named("gridsizes")=PDLI.nw_gridsizes,
    Named("lambda_poss")=PDLI.nw_lambda_poss,
    Named("nonzeros")=PDLI.nw_nonzeros,
    Named("lambdas")=PDLI.nw_lambdas,
    Named("nonzero_limits")=PDLI.nw_nonzero_limits,
    Named("criterion_values")=PDLI.nw_criterion_values,
    Named("SSRs")=PDLI.nw_SSRs,
    Named("grids")=PDLI.nw_grids,
    Named("residuals")=PDLI.nw_residuals,
    Named("opt_types")=PDLI.nw_opt_types,
    Named("selection_types")=PDLI.nw_selection_types,
    Named("nonzero_poss")=PDLI.nw_nonzero_poss
  );
  List inference=List::create(Named("Omegahat")=PDLI.Omegahat,
                              Named("R")=PDLI.R,
                              Named("q")=PDLI.q,
                              Named("standard_errors")=PDLI.std_errors_individual_vars,
                              Named("z_quantiles")=PDLI.z_quantiles,
                              Named("intervals")=PDLI.intervals,
                              Named("intervals_unscaled")=PDLI.intervals_unscaled,
                              Named("intervals_Rq")=PDLI.intervals_Rq,
                              Named("intervals_Rq_unscaled")=PDLI.intervals_Rq_unscaled,
                              Named("z_stats_Rq")=PDLI.z_stats_Rq,
                              Named("joint_chi2_stat")=PDLI.joint_chi2_stat,
                              Named("chi2_quantiles")=PDLI.chi2_quantiles
  );
  List EWC=List::create(Named("intervals_EWC")=PDLI.intervals_EWC,
                        Named("intervals_unscaled_EWC")=PDLI.intervals_unscaled_EWC,
                        Named("Omegahat_EWC")=PDLI.Omegahat_EWC
    );
  List NWfb=List::create(Named("intervals_NWfb")=PDLI.intervals_NWfb,
                        Named("intervals_unscaled_NWfb")=PDLI.intervals_unscaled_NWfb,
                        Named("Omegahat_NWfb")=PDLI.Omegahat_NWfb
  );
  return List::create(Named("misc")=misc,
                      Named("init")=init,
                      Named("nw")=nw,
                      Named("inference")=inference,
                      Named("bhat_1")=PDLI.bhat_1,
                      Named("bhat_1_unscaled")=PDLI.bhat_1_unscaled,
                      Named("gammahats")=PDLI.gammahats,
                      Named("Gammahat")=PDLI.Gammahat,
                      Named("Upsilonhat_inv")=PDLI.Upsilonhat_inv,
                      Named("Thetahat")=PDLI.Thetahat,
                      Named("EWC")=EWC,
                      Named("NWfb")=NWfb
  );
}






// //' Generate initial and nodewise lambda grids
// //' @param T_ number of rows in the X matrix
// //' @param N number of columns in the X matrix
// //' @param size number of lambdas in the grids
// //' @param X regressor matrix
// //' @param y dependent variable vector
// //' @param H indexes of relevant regressors
//[[Rcpp::export(.Rwrap_build_gridsXy)]]
List Rwrap_build_gridsXy(unsigned int T_, unsigned int N, unsigned int size, arma::mat X, arma::vec y, arma::uvec H, bool demean, bool scale){
  grids_output g=build_gridsXy(T_, N, size, X, y, H, demean, scale);
  return List::create(Named("init_grid")=g.init_grid,
                      Named("nw_grids")=g.nw_grids
  );
}

// LP_output local_projection(Nullable<NumericMatrix> r_, const arma::vec& x, const arma::vec& y, Nullable<NumericMatrix> q_,
//                            const bool& y_predetermined,const bool& cumulate_y, const unsigned int& hmax,
//                            const unsigned int& lags,const NumericVector& alphas, const bool& init_partial, const int& selection, const double& PIconstant,
//                            const bool& progress_bar, const unsigned int& threads){
//   arma::mat w;
//   arma::mat w_lags;
//   arma::mat regressors;
//   arma::vec dependent;
//   arma::mat joint_trimmed;
//   arma::mat dependent_trimmed;
//   arma::mat regressors_trimmed;
//   NumericMatrix trimmed_residuals;
//   List g;
//   List d;
//   List temp;
//   arma::mat tempmat;
//   NumericMatrix mThetahat;
//   NumericMatrix mUpsilonhat_inv;
//   NumericMatrix mnw_residuals;
//
//   List betahats(hmax+1);
//
//
//   arma::uvec H(1); H(0)=0;
//   LogicalVector nw_partials(1);nw_partials(0)=false;
//
//   int init_selection_type=selection;
//   arma::vec nw_selection_types(1); nw_selection_types(0)=false;
//
//   double init_nonzero_limit=0.5;
//   arma::vec nw_nonzero_limits(H.n_elem);
//   for(unsigned int i=0; i<H.n_elem; i++){
//     nw_nonzero_limits(i)=0.5;
//   }
//   double init_opt_threshold=1e-4;
//   arma::vec nw_opt_thresholds(H.n_elem);
//   for(unsigned int i=0; i<H.n_elem; i++){
//     nw_opt_thresholds(i)=1e-4;
//   }
//
//   int init_opt_type=3;
//
//   arma::vec nw_opt_types(1);nw_opt_types(0)=3;
//   arma::mat intervals(hmax+1,1+2*as<arma::mat>(alphas).n_elem, fill::zeros);
//   arma::mat R(1,1, fill::eye);
//   arma::vec Q(1, fill::ones);
//
//   unsigned int T_=y.n_elem;
//   arma::mat r;
//   if(r_.isNotNull()){
//     NumericMatrix r_temp(r_);
//     r=as<arma::mat>(r_temp);
//   }
//   arma::mat q;
//   if(q_.isNotNull()){
//     NumericMatrix q_temp(q_);
//     q=as<arma::mat>(q_temp);
//   }
//   bool is_same=false;
//   if(sum(abs(x-y))<1e-8){
//     is_same=true;
//   }
//   //estimate the LP at horizon 0 and save the nodewise parts
//   if(is_same){
//     w=join_horiz(r,x,q);//skip y in w, so we don't have two identical variables there
//   }else{
//     w=join_horiz(r,x,y,q);
//   }
//   w_lags=Rcpp_make_lags(w, lags);
//   regressors=join_horiz(x, r, w_lags);
//   if(is_same){//If x and y are the same, estimate at horizon 1. The only parts taken from this step will be the nodewise regressions.
//     dependent=na_matrix(T_,1);
//     if(cumulate_y){
//       for(unsigned int i=0; i<T_-1;i++){
//         dependent.row(i)=sum(y.rows(i,i+1));
//       }
//     }else{
//       for(unsigned int i=0; i<T_-1;i++){
//         dependent.row(i)=y.row(i+1);
//       }
//     }
//   }else{//If they're different, estimate horizon 0 normally.
//     dependent=y;
//   }
//   joint_trimmed=naomit(join_horiz(dependent,regressors));
//
//   dependent_trimmed=joint_trimmed.col(0);
//   regressors_trimmed=joint_trimmed.cols(1, joint_trimmed.n_cols-1);
//
//   g=Rwrap_build_gridsXy(regressors_trimmed.n_rows, regressors_trimmed.n_cols, 50, regressors_trimmed, dependent_trimmed, H, true, true);
//   d=Rwrap_partial_desparsified_lasso_inference(regressors_trimmed, dependent_trimmed, H, true, true, init_partial, nw_partials,
//                                                g["init_grid"], g["nw_grids"], init_selection_type, nw_selection_types,
//                                                init_nonzero_limit, nw_nonzero_limits, init_opt_threshold, nw_opt_thresholds, init_opt_type, nw_opt_types,
//                                                0, 0, alphas, R, Q, PIconstant, 0.05, progress_bar, threads,
//                                                R_NilValue, R_NilValue, R_NilValue);
//   mThetahat=as<NumericMatrix>(d["Thetahat"]);
//   mUpsilonhat_inv=as<NumericMatrix>(d["Upsilonhat_inv"]);
//   List temp2=d["nw"];
//   mnw_residuals=as<NumericMatrix>(temp2["residuals"]);
//
//   temp=d["inference"];
//   tempmat=as<arma::mat>(temp["intervals_unscaled"]);
//   temp=d["init"];
//   betahats(0)=temp["betahat"];
//
//   if(y_predetermined){
//     intervals.row(0)=arma::mat(1,1+2*as<arma::mat>(alphas).n_elem, fill::zeros);//if y is predetermined with respect to x, the response is 0 by assumption
//   }else if(is_same){
//     intervals.row(0)=arma::mat(1,1+2*as<arma::mat>(alphas).n_elem, fill::ones);//if x is the same as y, the response is 1 by assumption (only if no state dependence)
//   }else{
//     intervals.row(0)=tempmat.row(0);
//   }
//
//   //estimate at other horizons
//   Progress p(hmax, progress_bar);
//   for(unsigned int h=1; h<=hmax;h++){
//     if (Progress::check_abort()){
//       break;
//     }
//     dependent=na_matrix(T_,1);
//     if(cumulate_y){
//       for(unsigned int i=0; i<T_-h;i++){
//         dependent.row(i)=sum(y.rows(i,i+h));
//       }
//     }else{
//       for(unsigned int i=0; i<T_-h;i++){
//         dependent.row(i)=y.row(i+h);
//       }
//     }
//     joint_trimmed=naomit(join_horiz(dependent,regressors));
//     dependent_trimmed=joint_trimmed.col(0);
//     regressors_trimmed=joint_trimmed.cols(1, joint_trimmed.n_cols-1);
//     trimmed_residuals=mnw_residuals(Range(0,regressors_trimmed.n_rows-1), Range(0,0));
//     g=Rwrap_build_gridsXy(regressors_trimmed.n_rows, regressors_trimmed.n_cols, 50, regressors_trimmed, dependent_trimmed, H, true, true);
//     d=Rwrap_partial_desparsified_lasso_inference(regressors_trimmed, dependent_trimmed, H, true, true, init_partial, nw_partials,
//                                                  g["init_grid"], g["nw_grids"], init_selection_type, nw_selection_types,
//                                                  init_nonzero_limit, nw_nonzero_limits, init_opt_threshold, nw_opt_thresholds, init_opt_type, nw_opt_types,
//                                                  0, 0, alphas, R, Q, PIconstant, 0.05, false, 0,
//                                                  mThetahat, mUpsilonhat_inv,trimmed_residuals);
//     temp=d["inference"];
//     tempmat=as<arma::mat>(temp["intervals_unscaled"]);
//     intervals.row(h)=tempmat.row(0);
//     temp=d["init"];
//     betahats(h)=temp["betahat"];
//     p.increment();
//   }
//   LP_output LPo;
//   LPo.intervals=intervals;
//   LPo.manual_Thetahat=mThetahat;
//   LPo.betahats=betahats;
//   return LPo;
// }
//
// // [[Rcpp::export(.Rcpp_local_projection)]]
// List Rcpp_local_projection(Nullable<NumericMatrix> r_, const arma::vec& x, const arma::vec& y, Nullable<NumericMatrix> q_,
//                            const bool& y_predetermined,const bool& cumulate_y, const unsigned int& hmax,
//                            const unsigned int& lags,const NumericVector& alphas, const bool& init_partial, const int& selection, const double& PIconstant,
//                            const bool& progress_bar, const unsigned int& threads){
//   LP_output LPo=local_projection(r_, x, y,  q_,
//                                  y_predetermined, cumulate_y, hmax,
//                                  lags, alphas, init_partial, selection, PIconstant,
//                                  progress_bar, threads);
//   return(List::create(Named("intervals")=LPo.intervals,
//                       Named("manual_Thetahat")=LPo.manual_Thetahat,
//                       Named("betahats")=LPo.betahats));
// }


// [[Rcpp::export(.Rcpp_local_projection_state_dependent)]]
List Rcpp_local_projection_state_dependent(Nullable<NumericMatrix> r_, const arma::vec& x, const arma::vec& y, Nullable<NumericMatrix> q_, Nullable<NumericMatrix> state_dummy_,
                                           const bool& y_predetermined,const bool& cumulate_y, const unsigned int& hmax,
                                           const unsigned int& lags,const NumericVector& alphas, const bool& init_partial, const int& selection, const double& PIconstant,
                                           const bool& progress_bar, bool OLS, unsigned int threads, arma::mat seeds_mat){
  LP_state_dependent_output LPsdo=local_projection_state_dependent(r_, x, y, q_, state_dummy_,
                                                                   y_predetermined,cumulate_y, hmax,
                                                                   lags,alphas, init_partial, selection, PIconstant,
                                                                   progress_bar, OLS, threads, seeds_mat);
  return List::create(Named("intervals")=LPsdo.intervals,
                      Named("intervals_EWC")=LPsdo.intervals_EWC,////////////
                      Named("intervals_NWfb")=LPsdo.intervals_NWfb,//////////////
                      Named("manual_Thetahat")=LPsdo.manual_Thetahat,
                      Named("betahats")=LPsdo.betahats,
                      Named("regressors")=LPsdo.regressors,
                      Named("regressors_trimmed")=LPsdo.regressors_trimmed,
                      Named("init_residuals")=LPsdo.init_residuals,
                      Named("nw_residuals")=LPsdo.nw_residuals,
                      Named("dependents")=LPsdo.dependents);
}
