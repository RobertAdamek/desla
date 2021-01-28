#include <RcppArmadillo.h>
#include <math.h>
// [[Rcpp::depends(RcppArmadillo)]]
# include <omp.h>

using namespace Rcpp;
using namespace arma;
using namespace std;



// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

//struct definitions

struct lasso_output{
  unsigned int N, T, gridsize;
  arma::vec grid, y;
  arma::mat betahats, X;
  int opt_type; //1="naive", 2="covariance", 3="adaptive"
};

struct streamlined_lasso_output{
  arma::mat betahats;
};

struct partial_lasso_output{
  bool partial;
  unsigned int N, T, gridsize, h;
  arma::vec grid, y;
  arma::uvec H, minusH;
  arma::mat betahats, betahats_1, betahats_2, X, X_1, X_2;
  int opt_type; //1="naive", 2="covariance", 3="adaptive"
};

struct streamlined_partial_lasso_output{
  arma::mat betahats;
};

struct lasso_selected_output{
  unsigned int N, T, gridsize, lambda_pos, nonzero;
  double lambda, nonzero_limit, criterion_value, SSR;
  arma::vec grid, y, residual, betahat;
  arma::mat X;
  int opt_type; //1="naive", 2="covariance", 3="adaptive"
  int selection_type; //1="BIC", 2="AIC", 3="EBIC"
};

struct streamlined_lasso_selected_output{
  unsigned int lambda_pos, nonzero;
  double lambda, SSR;
  arma::vec  residual, betahat;
};

struct partial_lasso_selected_output{
  bool partial;
  unsigned int N, T, gridsize, lambda_pos, nonzero, h;
  double lambda, nonzero_limit, criterion_value, SSR;
  arma::vec grid, y, residual, betahat, betahat_1, betahat_2;
  arma::uvec H, minusH;
  arma::mat X, X_1, X_2;
  int opt_type; //1="naive", 2="covariance", 3="adaptive"
  int selection_type; //1="BIC", 2="AIC", 3="EBIC"
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
};

struct streamlined_selection_output{
  unsigned int lambda_pos, nonzero;
  double lambda,  SSR;
  arma::vec betahat, residual;
};

struct partial_desparsified_lasso_output{
  bool init_partial;
  LogicalVector nw_partials;
  unsigned int N, T, h;
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
};



struct partial_desparsified_lasso_inference_output{
  bool init_partial;
  LogicalVector nw_partials;
  unsigned int N, T, h;
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
  arma::mat Omegahat;
  arma::mat R;
  arma::vec q;
  arma::vec z_quantiles;
  arma::mat intervals;
  double joint_chi2_stat;
  arma::vec chi2_quantiles;
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

//auxilliary functions

selection_output selectBIC(const arma::mat& betahats, const arma::mat& X, const arma::vec& y, const arma::vec& grid, const unsigned int& N, const unsigned int& T, const unsigned int& gridsize, const double& nonzero_limit){
  double c, best_criterion_value=std::numeric_limits<double>::infinity(), best_SSR=0;
  unsigned int lambda_pos=0, nonzero=0, best_nonzero=0;
  arma::vec betahat(N);
  arma::vec residual(T), best_residual(T);
  double SSR=0;
  for(unsigned int i=0;i<gridsize;i++){
    betahat=betahats.col(i);
    nonzero=std::count_if(betahat.begin(), betahat.end(), [](double j){return j!=0;});
    residual=y-X*betahat;
    SSR=as_scalar(residual.t()*residual);
    c=T*log(SSR/double(T))+log(T)*nonzero;
    if(c < best_criterion_value && nonzero<=T*nonzero_limit){
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
  return ret;
}

selection_output selectAIC(const arma::mat& betahats, const arma::mat& X, const arma::vec& y, const arma::vec& grid, const unsigned int& N, const unsigned int& T, const unsigned int& gridsize, const double& nonzero_limit){
  double c, best_criterion_value=std::numeric_limits<double>::infinity(), best_SSR=0;
  unsigned int lambda_pos=0, nonzero=0, best_nonzero=0;
  arma::vec betahat(N);
  arma::vec residual(T), best_residual(T);
  double SSR=0;
  for(unsigned int i=0;i<gridsize;i++){
    betahat=betahats.col(i);
    nonzero=std::count_if(betahat.begin(), betahat.end(), [](double j){return j!=0;});
    residual=y-X*betahat;
    SSR=as_scalar(residual.t()*residual);
    c=T*log(SSR/double(T))+2.0*nonzero;
    if(c < best_criterion_value && nonzero<=T*nonzero_limit){
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
  return ret;
}

selection_output selectEBIC(const arma::mat& betahats, const arma::mat& X, const arma::vec& y, const arma::vec& grid, const unsigned int& N, const unsigned int& T, const unsigned int& gridsize, const double& nonzero_limit){
  double c, best_criterion_value=std::numeric_limits<double>::infinity(), best_SSR=0;
  unsigned int lambda_pos=0, nonzero=0, best_nonzero=0;
  arma::vec betahat(N);
  arma::vec residual(T), best_residual(T);
  double SSR=0;
  double gamma=1.0;
  for(unsigned int i=0;i<gridsize;i++){
    betahat=betahats.col(i);
    nonzero=std::count_if(betahat.begin(), betahat.end(), [](double j){return j!=0;});
    residual=y-X*betahat;
    SSR=as_scalar(residual.t()*residual);
    c=T*log(SSR/double(T))+log(T)*nonzero+2.0*nonzero*gamma*log(N);
    if(c < best_criterion_value && nonzero<=T*nonzero_limit){
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

grids_output build_gridsXy(const unsigned int& T, const unsigned int N, const unsigned int& size, const arma::mat& X, const arma::vec& y, const arma::uvec& H){
  unsigned int h=H.n_elem;
  unsigned int j,i;
  arma::vec X_j;
  arma::uvec index_minusj;
  arma::mat X_minusj;
  double lambda_max=max(abs(X.t()*y)/double(T));
  arma::vec init_grid=buildgrid(size,lambda_max,1.0/double(T*10.0));
  arma::mat nw_grids(h, size);
  for(i=0; i<h; i++){
    j=H(i);
    X_j=X.col(j);
    index_minusj=linspace<arma::uvec>(0,N-1,N); index_minusj.shed_row(j);
    X_minusj=X.cols(index_minusj);
    lambda_max=max(abs(X_minusj.t()*X_j)/double(T));
    nw_grids.row(i)=buildgrid(size,lambda_max,1.0/double(T*10.0)).t();
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

arma::mat coordinate_descent_naive(const arma::mat& X, const arma::colvec& y, const arma::vec& grid, const double& opt_threshold,
                             const unsigned int& N, const unsigned int& T, const unsigned int& gridsize){
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
  arma::vec x_j(T);
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
      betahat(j)=soft_threshold(as_scalar(x_j.t()*partial_res)/x_crossprods(j), T*grid(g)/x_crossprods(j));
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
        betahat(j)=soft_threshold(as_scalar(x_j.t()*partial_res)/x_crossprods(j), T*grid(g)/x_crossprods(j));
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
                                  const unsigned int& N, const unsigned int& T, const unsigned int& gridsize){
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
  arma::vec x_j(T);
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
      betahat(j)=soft_threshold((xy(j)-x_jx_kbeta_k)/x_crossprods(j,j), T*grid(g)/x_crossprods(j,j));
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
        betahat(j)=soft_threshold((xy(j)-x_jx_kbeta_k)/x_crossprods(j,j), T*grid(g)/x_crossprods(j,j));
      }
      change=sqrt(as_scalar((betahat-betahat_old).t()*(betahat-betahat_old)))/N;
      betahat_old=betahat;
    }
    betahat_mat.col(g)=betahat;
  }
  return betahat_mat;
}

arma::mat LRVestimator(const arma::vec& init_residual, const arma::mat& nw_residuals, const unsigned int& N, const unsigned int& T, const unsigned int& h, const double& LRVtrunc, const double& T_multiplier){
  arma::mat Omegahat(h,h,fill::zeros);
  unsigned int Q_T=ceil(pow(T_multiplier*double(T),LRVtrunc));
  arma::vec uhat=init_residual;
  arma::mat vhat=nw_residuals;
  arma::mat What(T,h);
  for(unsigned int a=0;a<h;a++){
    What.col(a)=uhat%vhat.col(a);
  }
  arma::vec Whatbar=mean(What,0).as_col();
  //this is not memory efficient, I don't need all the xi_l^(j,k) at once
  cube xi(h,h,Q_T);
  for(unsigned int a=0;a<h;a++){
    for(unsigned int b=0;b<h;b++){
      for(unsigned int l=0;l<=Q_T-1;l++){
        //I think I need to keep redefining them since their sizes change
        arma::vec W_j=What.submat(l,a,(T-1),a);
        arma::vec W_kminusl=What.submat(0,b,(T-l-1),b);
        xi(a,b,l)=as_scalar( (W_j-as_scalar(Whatbar[a])).t()*(W_kminusl-as_scalar(Whatbar[b])) )/(T-l);
      }
    }
  }
  //build the lower triangular part of Omegahat
  arma::vec kernel(Q_T);
  kernel(0)=0; //sort out the calculation of the terms for l>=1 with a loop, because the first element is treated differently
  if(Q_T>1){
    for(double l=1;l<Q_T;l++){
      kernel(l)=1.0-double(l)/double(Q_T);
    }
  }
  for(unsigned int a=0;a<h;a++){
    for(unsigned int b=0;b<=a;b++){
      arma::vec xijk=xi.tube(a,b);
      arma::vec xikj=xi.tube(b,a);
      Omegahat(b,a)=Omegahat(a,b)=xijk(0)+as_scalar( kernel.t()*(xijk+xikj) );
    }
  }
  return Omegahat;
}




//main functions

lasso_output lasso(const arma::mat& X, const arma::colvec& y, const arma::vec& grid,
                   const double& opt_threshold, const int& opt_type){
  unsigned int T=X.n_rows;
  unsigned int N=X.n_cols;
  unsigned int gridsize=grid.n_elem;
  arma::mat betahats(N,gridsize);

  switch(opt_type) {
  case 1: //"naive"
    betahats=coordinate_descent_naive(X, y, grid, opt_threshold,
                                      N, T, gridsize);
    break;
  case 2: //"covariance"
    betahats=coordinate_descent_covariance(X, y, grid, opt_threshold,
                                           N, T, gridsize);
    break;
  case 3: //"adaptive"
    if(N>T){
      betahats=coordinate_descent_naive(X, y, grid, opt_threshold,
                                        N, T, gridsize);
    }
    else{
      betahats=coordinate_descent_covariance(X, y, grid, opt_threshold,
                                             N, T, gridsize);
    }
    break;
  default:
    stop("Warning: Invalid opt_type");
  }
  lasso_output ret;
  ret.N=N;
  ret.T=T;
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
  unsigned int T=X.n_rows;
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
    arma::mat M_X1=(mat(T,T,fill::eye)-X_1*X1X1inv*X_1.t());
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
  ret.T=T;
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

lasso_selected_output lasso_selected(const arma::mat& X, const arma::colvec& y, const arma::vec& grid, const int& selection_type, const double& nonzero_limit,
                                     const double& opt_threshold, const int& opt_type){
  lasso_output L=lasso(X, y, grid, opt_threshold, opt_type);
  selection_output S;
  switch(selection_type) {
  case 1: //"BIC"
    S=selectBIC(L.betahats, X, y, grid, L.N, L.T, L.gridsize,nonzero_limit);
    break;
  case 2: //"AIC"
    S=selectAIC(L.betahats, X, y, grid, L.N, L.T, L.gridsize,nonzero_limit);
    break;
  case 3: //"EBIC"
    S=selectEBIC(L.betahats, X, y, grid, L.N, L.T, L.gridsize,nonzero_limit);
    break;
  default:
    stop("Warning: Invalid selection_type");
  }
  lasso_selected_output ret;
  ret.N=L.N;
  ret.T=L.T;
  ret.gridsize=L.gridsize;
  ret.lambda_pos=S.lambda_pos;
  ret.nonzero=S.nonzero;
  ret.lambda=S.lambda;
  ret.nonzero_limit=S.nonzero_limit;
  ret.criterion_value=S.criterion_value;
  ret.SSR=S.SSR;
  ret.grid=L.grid;
  ret.y=L.y;
  ret.residual=S.residual;
  ret.betahat=S.betahat;
  ret.X=L.X;
  ret.opt_type=L.opt_type;
  ret.selection_type=S.selection_type;
  return ret;
}

partial_lasso_selected_output partial_lasso_selected(const arma::mat& X, const arma::colvec& y, const arma::uvec& H, const bool& partial, const arma::vec& grid, const int& selection_type, const double& nonzero_limit,
                                                     const double& opt_threshold, const int& opt_type){
  partial_lasso_output PL=partial_lasso(X, y, H, partial, grid, opt_threshold, opt_type);
  selection_output S;
  switch(selection_type) {
  case 1: //"BIC"
    S=selectBIC(PL.betahats, X, y, grid, PL.N, PL.T, PL.gridsize,nonzero_limit);
    break;
  case 2: //"AIC"
    S=selectAIC(PL.betahats, X, y, grid, PL.N, PL.T, PL.gridsize,nonzero_limit);
    break;
  case 3: //"EBIC"
    S=selectEBIC(PL.betahats, X, y, grid, PL.N, PL.T, PL.gridsize,nonzero_limit);
    break;
  default:
    stop("Warning: Invalid selection_type");
  }
  partial_lasso_selected_output ret;
  ret.partial=partial;
  ret.N=PL.N;
  ret.T=PL.T;
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
  ret.betahat_1=PL.betahats_1.col(S.lambda_pos);
  ret.betahat_2=PL.betahats_2.col(S.lambda_pos);
  ret.H=PL.H;
  ret.minusH=PL.minusH;
  ret.X=PL.X;
  ret.X_1=PL.X_1;
  ret.X_2=PL.X_2;
  ret.opt_type=PL.opt_type;
  ret.selection_type=S.selection_type;
  return ret;
}

partial_desparsified_lasso_output partial_desparsified_lasso(const arma::mat& X, const arma::colvec& y, const arma::uvec& H, const bool& init_partial, const LogicalVector& nw_partials, const arma::vec& init_grid, const arma::mat& nw_grids, const int& init_selection_type, const arma::vec& nw_selection_types,
                                                             const double& init_nonzero_limit, const arma::vec& nw_nonzero_limits, const double& init_opt_threshold, const arma::vec& nw_opt_thresholds, const int& init_opt_type, const arma::vec& nw_opt_types){


  partial_lasso_selected_output init_L=partial_lasso_selected(X, y, H, init_partial, init_grid, init_selection_type, init_nonzero_limit,
                                                              init_opt_threshold, init_opt_type);
  unsigned int h=init_L.h;
  unsigned int N=init_L.N;
  unsigned int T=init_L.T;
  unsigned int j, i;
  arma::uvec i_;
  bool nw_partial;
  int nw_selection_type, nw_opt_type;
  double nw_nonzero_limit, nw_opt_threshold;
  double tauhat_j;
  arma::vec bhat_1(h), x_j(T), gammahat_j, nw_grid;
  arma::vec nw_gridsizes(h), nw_lambdas(h), nw_criterion_values(h), nw_SSRs(h), nw_lambda_poss(h), nw_nonzeros(h);
  arma::uvec minusj, Hminusj, nw_H;
  arma::mat nw_residuals(T,h), Thetahat(h,N, fill::zeros), Upsilonhat_inv(h,h, fill::zeros), gammahats(N-1,h), Gammahat(h,N, fill::zeros), Xminusj(T,N-1);
  partial_lasso_selected_output nw_L;
  for(i=0;i<h;i++){
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
                                nw_opt_threshold, nw_opt_type);
    tauhat_j=nw_L.SSR/double(T)+2*nw_L.lambda*sum(abs(nw_L.betahat));
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
  }
  Thetahat=Upsilonhat_inv*Gammahat;
  bhat_1=init_L.betahat_1+Thetahat*X.t()*(init_L.residual)/double(T);

  partial_desparsified_lasso_output ret;
  ret.init_partial=init_partial;
  ret.nw_partials=nw_partials;
  ret.N=init_L.N;
  ret.T=init_L.T;
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
  return ret;
}

partial_desparsified_lasso_inference_output partial_desparsified_lasso_inference(const arma::mat& X, const arma::colvec& y, const arma::uvec& H, const bool& init_partial, const LogicalVector& nw_partials, const arma::vec& init_grid, const arma::mat& nw_grids, const int& init_selection_type, const arma::vec& nw_selection_types,
                                                                                 const double& init_nonzero_limit, const arma::vec& nw_nonzero_limits, const double& init_opt_threshold, const arma::vec& nw_opt_thresholds, const int& init_opt_type, const arma::vec& nw_opt_types,
                                                                                 const double& LRVtrunc, const double& T_multiplier, const NumericVector& alphas, const arma::mat& R, const arma::vec& q){
  partial_desparsified_lasso_output PDL=partial_desparsified_lasso(X, y, H, init_partial, nw_partials, init_grid, nw_grids, init_selection_type, nw_selection_types,
                                                                   init_nonzero_limit, nw_nonzero_limits, init_opt_threshold, nw_opt_thresholds, init_opt_type, nw_opt_types);
  arma::mat Omegahat=LRVestimator(PDL.init_residual, PDL.nw_residuals, PDL.N, PDL.T, PDL.h, LRVtrunc, T_multiplier);
  arma::vec z_quantiles=qnorm(alphas/2.0,0.0,1.0,false,false);

  unsigned int P=R.n_rows;
  arma::vec chi2_quantiles=qchisq(alphas,P,false,false);//add the correct degrees of freedom
  arma::mat covariance=PDL.Upsilonhat_inv*Omegahat*PDL.Upsilonhat_inv;
  arma::vec Rbhat_1=R*PDL.bhat_1;
  arma::vec std_errors(P);
  for(unsigned int p=0; p<P; p++){
    std_errors(p)=sqrt( as_scalar(R.row(p)*covariance*(R.row(p)).t())/double(PDL.T) );
  }
  arma::mat intervals(P, 2*alphas.length()+1);
  intervals.col(alphas.length())=Rbhat_1;
  for(unsigned int p=0; p<P; p++){
    for(unsigned int j=0; j<alphas.length(); j++){
      intervals(p,j)=Rbhat_1(p)-z_quantiles(j)*std_errors(p);
      intervals(p, 2*alphas.length()-j)=Rbhat_1(p)+z_quantiles(j)*std_errors(p);
    }
  }
  double joint_chi2_stat=as_scalar( (Rbhat_1-q).t()*inv_sympd(R*covariance*R.t()/double(PDL.T))*(Rbhat_1-q) );

  partial_desparsified_lasso_inference_output ret;

  ret.init_partial=PDL.init_partial;
  ret.nw_partials=PDL.nw_partials;
  ret.N=PDL.N;
  ret.T=PDL.T;
  ret.h=PDL.h;
  ret.init_gridsize=PDL.init_gridsize;
  ret.init_lambda_pos=PDL.init_lambda_pos;
  ret.init_nonzero=PDL.init_nonzero;
  ret.init_lambda=PDL.init_lambda;
  ret.init_nonzero_limit=PDL.init_nonzero_limit;
  ret.init_criterion_value=PDL.init_criterion_value;
  ret.init_SSR=PDL.init_SSR;
  ret.bhat_1=PDL.bhat_1;
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
  ret.z_quantiles=z_quantiles;
  ret.intervals=intervals;
  ret.joint_chi2_stat=joint_chi2_stat;
  ret.chi2_quantiles=chi2_quantiles;
  return ret;
}




//wrapper functions

//' Calculate the desparsified lasso
//' @param X regressor matrix
//' @param y dependent variable vector
//' @param H indexes of relevant regressors
//' @param init_partial (optional) boolean, true if you want the initial lasso to be partially penalized (false by default)
//' @param nw_partials (optional) boolean vector with the dimension of H, trues if you want the nodewise regressions to be partially penalized (all false by default)
//' @param gridsize (optional) integer, how many different lambdas there should be in both inital and nodewise grids (100 by default)
//' @param init_grid (optional) vector, containing user specified initial grid
//' @param nw_grids (optional) matrix with number of rows the size of H, rows containing user specified grids for the nodewise regressions
//' @param init_selection_type (optional) integer, how should lambda be selected in the inital regression, 1=BIC, 2=AIC, 3=EBIC (1 by default)
//' @param nw_selection_types (optional) inteter vector with the dimension of H, how should lambda be selected in the nodewise regressions, 1=BIC, 2=AIC, 3=EBIC (all 1s by default)
//' @param init_nonzero_limit (optional) number controlling the maximum number of nonzeros that can be selected in the initial regression (0.5 by default, meaning no more than 0.5*T regressors can have nonzero estimates)
//' @param nw_nonzero_limits (optional) vector with the dimension of H, controlling the maximum number of nonzeros that can be selected in the nodewise regressions (0.5s by default)
//' @param init_opt_threshold (optional) optimization threshold for the coordinate descent algorithm in the inital regression (10^(-4) by default)
//' @param nw_opt_thresholds (optional) vector with the dimension of H, optimization thresholds for the coordinate descent algorithm in the nodewise lasso regression (10^(-4)s by default)
//' @param init_opt_type (optional) integer, which type of coordinate descent algorithm should be used in the inial regression, 1=naive, 2=covariance, 3=adaptive (3 by default)
//' @param nw_opt_types (optional)inteter vector with the dimension of H, which type of coordinate descent algorithm should be used in the nodewise regressions, 1=naive, 2=covariance, 3=adaptive (3s by default)
//' @param LRVtrunc (optional) parameter controlling the bandwidth Q_T used in the long run covariance matrix, Q_T=ceil(T_multiplier*T^LRVtrunc) (LRVtrunc=0.2 by default)
//' @param T_multiplier (optional) parameter controlling the bandwidth Q_T used in the long run covariance matrix, Q_T=ceil(T_multiplier*T^LRVtrunc) (Tmultiplier=2 by default)
//' @param alphas (optional) vector of significance levels (c(0.01,0.05,0.1) by default)
//' @param R (optional) matrix with number of columns the dimension of H, used to test the null hypothesis R*beta=q (identity matrix as default)
//' @param q (optional) vector of size same as the rows of H, used to test the null hypothesis R*beta=q (zeroes by default)
// [[Rcpp::export]]
List Rwrap_partial_desparsified_lasso_inference(arma::mat& X, arma::colvec& y, arma::uvec& H, bool& init_partial, LogicalVector& nw_partials, arma::vec& init_grid, arma::mat& nw_grids, int& init_selection_type, arma::vec& nw_selection_types,
                                                double& init_nonzero_limit, arma::vec& nw_nonzero_limits, double& init_opt_threshold, arma::vec& nw_opt_thresholds, int& init_opt_type, arma::vec& nw_opt_types,
                                                double& LRVtrunc, double& T_multiplier, NumericVector& alphas, arma::mat& R, arma::vec& q){
  partial_desparsified_lasso_inference_output PDLI=partial_desparsified_lasso_inference(X, y, H, init_partial, nw_partials, init_grid, nw_grids, init_selection_type, nw_selection_types,
                                                                                        init_nonzero_limit, nw_nonzero_limits, init_opt_threshold, nw_opt_thresholds, init_opt_type, nw_opt_types,
                                                                                        LRVtrunc, T_multiplier, alphas, R, q);
  List misc=List::create(Named("N")=PDLI.N,
                         Named("T")=PDLI.T,
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
                         Named("selection_type")=PDLI.init_selection_type
  );
  List nw=List::create(Named("partials")=PDLI.nw_partials,
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
                       Named("selection_types")=PDLI.nw_selection_types
  );
  List inference=List::create(Named("Omegahat")=PDLI.Omegahat,
                              Named("R")=PDLI.R,
                              Named("q")=PDLI.q,
                              Named("z_quantiles")=PDLI.z_quantiles,
                              Named("intervals")=PDLI.intervals,
                              Named("joint_chi2_stat")=PDLI.joint_chi2_stat,
                              Named("chi2_quantiles")=PDLI.chi2_quantiles
  );
  return List::create(Named("misc")=misc,
                      Named("init")=init,
                      Named("nw")=nw,
                      Named("inference")=inference,
                      Named("bhat_1")=PDLI.bhat_1,
                      Named("gammahats")=PDLI.gammahats,
                      Named("Gammahat")=PDLI.Gammahat,
                      Named("Upsilonhat_inv")=PDLI.Upsilonhat_inv,
                      Named("Thetahat")=PDLI.Thetahat
  );
}



//' Generate inital and nodewise lambda grids
//' @param T number of rows in the X matrix
//' @param N number of columns in the X matrix
//' @param size number of lambdas in the grids
//' @param X regressor matrix
//' @param y dependent variable vector
//' @param H indexes of relevant regressors
//[[Rcpp::export]]
List Rwrap_build_gridsXy(unsigned int& T, unsigned int N, unsigned int& size, arma::mat& X, arma::vec& y, arma::uvec& H){
  grids_output g=build_gridsXy(T, N, size, X, y, H);
  return List::create(Named("init_grid")=g.init_grid,
                      Named("nw_grids")=g.nw_grids
  );
}



