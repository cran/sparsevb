#ifndef FIT_LOGIT_H
#define FIT_LOGIT_H

#include <RcppEnsmallen.h>
#include <cmath>
#include <string>
#include <vector>

#include "laplace_obj_fn.h"
#include "misc_fn.h"

// [[Rcpp::depends(RcppEnsmallen)]]

// [[Rcpp::export]]
Rcpp::List fit_logistic(const arma::mat &X,
                        const arma::vec &Y,
                        arma::vec mu,
                        arma::vec sigma,
                        arma::vec gamma,
                        const double &alpha,
                        const double &beta,
                        const double &lambda,
                        const arma::uvec &update_order,
                        const std::string &prior,
                        const size_t &max_iter,
                        const double &tol);

#endif
