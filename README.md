# desla: Desparsified Lasso inference for Time Series

## Installation
### CRAN
The easiest way to install the package from CRAN is with the command:
``` r
install.packages("desla")
```
### GitHub
Working versions of updates to the package are available on GitHub. These can be installed easily with the devtools package:
``` r
install.packages("devtools")
library(devtools)
```
You can then install the desla package from the GitHub depository directly by running:
``` r
devtools::install_github("RobertAdamek/desla")
```
## Load Package
After installation, the package can be loaded in the standard way:
``` r
library(desla)
```
## Usage
The following toy example demonstrates how to use the functions:
``` r
X<-matrix(rnorm(100*100), nrow=100)
y<-X[,1:4] %*% c(1, 2, 3, 4) + rnorm(100)
```
The desla function provides inference on parameters of the linear regression of \code{y} on \code{X}, using the desparsified lasso as detailed in Adamek et al. (2020a).
First, we specify the indices of variables for which confidence intervals are required via the argument \code{H}.
The desla function then simply takes, the response \code{y}, the predictor matrix \code{X} and the set of indices \code{H} as input.
Finally, the output slot \code{intervals} displays the parameter estimates of all variables indexed by H, together with the 
lower and upper bounds of the confidence intervals for the signifiance levels 0.01,0.05, and 0.1 (default).
```r
H<-1:10
d<-desla(X=X, y=y, H=H)
d$intervals
```
For optional arguments and other details, see the function documentation with the command:
``` r
?desla
```
The second key function of the package is the HDLP function which implements the high-dimensional local projections detailed in Adamek et al. (2022b).
As an example, consider the response of \code{y} to a shock in the fourth predictor variable.
We allow all other predictor variables to react within the same period to the shock (by including them in the argument \code{q}) and simply use one lag in the LP model. Then the follow lines of code obtain the impulse responses. By applying the 
plot function to the output of the HDLP function, the corresponding impulse response function can be visualized.
```r
h<-HDLP(x=X[,4], y=y, q=X[,-4], hmax=5, lags=1)
plot(h)
```
The function also implements the state-based local projections of Ramey & Zubairy (2018) with the optional \code{stat_variables} argument.
This time, two separate impulse response functions are obtained for the response of \code{y} to a 
shock in the fourth predictor: one for state A and one for state B.
```r
s<-matrix(c(rep(1,50),rep(0,100),rep(1,50)), ncol=2, dimnames = list(NULL, c("A","B")))
h_s<-HDLP(x=X[,4], y=y, q=X[,-4], state_variables=s, hmax=5, lags=1)
plot(h_s)
```
For other optional arguments and details, see the function documentation with the command:
``` r
?HDLP
```

## References
Adamek, R., S. Smeekes, and I. Wilms (2022a). Lasso inference for high-dimensional time series.
*Journal of Econometrics*, Forthcoming.

Adamek, R., S. Smeekes, and I. Wilms (2022b). Local Projection inference in High Dimensions. arXiv e-print 2209.03218.

Ramey, V. A. and S. Zubairy (2018). Government spending multipliers in good times and in bad:
evidence from US historical data. *Journal of Political Economy 126*, 850–901.
