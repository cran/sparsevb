#ifndef LAPLACE_OBJ_FN_H
#define LAPLACE_OBJ_FN_H

#define _USE_MATH_DEFINES

#include <RcppArmadillo.h>

#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]

class laplace_obj_fn {
private:
  const double coef_sq;
  const double coef_lin;
  const double lambda;

public:
  laplace_obj_fn(const double &coef_sq, const double &coef_lin,
                 const double &lambda);

  double EvaluateWithGradient(const arma::mat &x, arma::mat &gradient);
};

#endif
