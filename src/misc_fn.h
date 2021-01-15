#ifndef MISC_FN_H
#define MISC_FN_H

#include <RcppArmadillo.h>

#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]

arma::vec entropy(const arma::vec &x);

double sigmoid(const double &x);

arma::vec gram_diag(const arma::mat &X);

#endif
