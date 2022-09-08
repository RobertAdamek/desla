# desla 0.2.0

* The function HDLP_state_dependent is removed, and its functionality incorporated as optional arguments into the function HDLP.

* Added a plot function which creates plots of impulse responses based on the objects created by the HDLP function.  

* The desla function is now parallelized over the nodewise regressions, and displays a progress bar. 

* Various fixes to prevent crashes.

# desla 0.1.0.9000

* Added the function HDLP_state_dependent, which allows for estimation of impulse responses in different states, and also an option to estimate them with OLS rather than desparsified lasso, using the same long-run variance estimator.

* Fixed bug in the long run covariance function, which occasionally caused crashes

# desla 0.1.0 (commit 3b0833b)

* First published version of the package
