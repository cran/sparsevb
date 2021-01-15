#include "laplace_obj_fn.h"

laplace_obj_fn::laplace_obj_fn(const double &coef_sq, const double &coef_lin,
                               const double &lambda)
    : coef_sq(coef_sq), coef_lin(coef_lin), lambda(lambda) {}

double laplace_obj_fn::EvaluateWithGradient(const arma::mat &x,
                                            arma::mat &gradient) {
  double mu = x(0);
  // double sigma = std::abs(x(1));
  double sigma = x(1);

  double exp_term =
      lambda * M_SQRT1_2 * M_2_SQRTPI * std::exp(-0.5 * pow(mu / sigma, 2));
  double erf_term = lambda * std::erf(M_SQRT1_2 * mu / sigma);

  gradient(0) = 2 * coef_sq * mu + erf_term + coef_lin;
  gradient(1) = 2 * coef_sq * sigma + exp_term - 1 / sigma;

  return (erf_term + coef_lin) * mu +
         coef_sq * (std::pow(mu, 2) + std::pow(sigma, 2)) -
         std::log(std::abs(sigma)) + sigma * exp_term;
}
