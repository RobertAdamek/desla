# desla: Desparsified Lasso inference for Time Series

## Installation and Loading

### Installation
To make use of this package, the easiest approach is to first install the devtools package. This can be done by running the following commands in the R console:
``` r
install.packages("devtools")
library(devtools)
```
You can then install the desla package from the GitHub depository by running:
``` r
devtools::install_github("RobertAdamek/desla")
```
### Load Package
After installation, the package can be loaded in the standard way:
``` r
library(desla)
```
## Usage
The package contains one function (also called "desla"). For details on how to use it, you can access the function documentation with the command:
``` r
?desla
```
An example of how the desla function can be used:
``` r
X<-matrix(rnorm(100*100), nrow=100)
y<-X[,1:4] %*% c(1, 2, 3, 4) + rnorm(100)
H<-c(1, 2, 3, 4)
d<-desla(X, y, H)
```

