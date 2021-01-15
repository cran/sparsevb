#include "fit_logit.h"

Rcpp::List fit_logistic(const arma::mat &X, const arma::vec &Y, arma::vec mu,
                        arma::vec sigma, arma::vec gamma, const double &alpha,
                        const double &beta, const double &lambda,
                        const arma::uvec &update_order,
                        const std::string &prior, const size_t &max_iter,
                        const double &tol) {
  // initialize entropy loss function
  arma::vec old_entr = entropy(gamma);

  // used to store successive variational bounds
  double old_bound = 0;
  std::vector<double> varbound;

  // pre-process update parameters
  arma::vec approx_mean = gamma % mu;
  arma::vec X_appm = X * approx_mean;
  arma::rowvec YX_vec = (Y - 0.5).t() * X;
  arma::vec eta(Y.n_elem, arma::fill::ones);
  arma::vec eta_hyp = 0.25 * arma::tanh(0.5 * eta) / eta;

  double const_lodds = std::log(alpha) - std::log(beta);
  if (prior == "laplace") {
    const_lodds += std::log(lambda) + 0.5;

  } else if (prior == "gaussian") {
    const_lodds -= std::log(lambda);
  }

  // iteration loop
  for (size_t i = 0; i < max_iter; ++i) {

    // pre-processing per iteration
    arma::rowvec coef_sq = eta_hyp.t() * arma::square(X);

    if (prior == "gaussian") {
      // implements equation (25) of Carbonetto et al.
      sigma = 1 / arma::sqrt(2 * coef_sq.t() + 1 / std::pow(lambda, 2));
    }

    // coordinate update loop
    for (arma::uword k = 0; k < mu.n_elem; ++k) {
      // check if interrupt signal was sent from R
      Rcpp::checkUserInterrupt();

      // the current update dimension
      arma::uword j = update_order(k);

      // delete the j-th column from running sum
      X_appm -= approx_mean(j) * X.col(j);

      if (prior == "laplace") {
        // initialize L-BFGS optimizer from ensmallen
        ens::L_BFGS optim;

        // start optimization at previous value
        arma::mat x(2, 1);
        x(0) = mu(j);
        x(1) = sigma(j);

        // implements equation (11) of the paper
        laplace_obj_fn f(coef_sq(j),
                         2 * arma::dot(eta_hyp % X.col(j), X_appm) - YX_vec(j),
                         lambda);

        // optimize and save function value
        double opt = optim.Optimize(f, x);
        mu(j) = x(0);
        // sigma(j) = std::abs(x(1));
        sigma(j) = x(1);

        // implements equation (12) of the paper
        if (j > 0) {
          gamma(j) = sigmoid(const_lodds - opt);
        }

      } else if (prior == "gaussian") {
        // implements equation (26) of Carbonetto et al.
        mu(j) = std::pow(sigma(j), 2) *
                (YX_vec(j) - 2 * arma::dot(eta_hyp % X.col(j), X_appm));

        // implements equation (27) of Carbonetto et al.
        if (j > 0) {
          gamma(j) = sigmoid(const_lodds + std::log(sigma(j)) +
                           0.5 * std::pow(mu(j) / sigma(j), 2));
        }
      }

      // add j-th column with updated values
      approx_mean(j) = gamma(j) * mu(j);
      X_appm += approx_mean(j) * X.col(j);
    }

    // implements equation (32) of the paper
    eta = arma::sqrt(
        arma::square(X) * (gamma % (arma::square(mu) + arma::square(sigma))) +
        arma::square(X_appm) - arma::square(X) * arma::square(approx_mean));
    eta_hyp = 0.25 * arma::tanh(0.5 * eta) / eta;

    // update variational bound
    double new_bound =
        arma::accu(arma::log1p(arma::exp(-eta)) - 0.5 * eta) +
        arma::dot(YX_vec, approx_mean) -
        arma::dot(gamma, arma::log(gamma + arma::datum::eps) - std::log(alpha) +
                             std::log(alpha + beta)) -
        arma::dot(1 - gamma, arma::log(1 - gamma + arma::datum::eps) -
                                 std::log(beta) + std::log(alpha + beta));

    // Rcpp::Rcout << "1: " << arma::accu(arma::log1p(arma::exp(-eta)) - 0.5 *
    // eta + arma::square(eta) % eta_hyp) << "\n"; Rcpp::Rcout << "2: " <<
    // arma::dot(X_appm, eta_hyp % X_appm) << "\n"; Rcpp::Rcout << "3: " <<
    // arma::dot(YX_vec, approx_mean) << "\n"; Rcpp::Rcout << "4: " <<
    // arma::dot(gamma, arma::log(gamma + arma::datum::eps) - std::log(alpha) +
    // std::log(alpha + beta)) << "\n"; Rcpp::Rcout << "5: " << arma::dot(1 -
    // gamma, arma::log(1 - gamma + arma::datum::eps) - std::log(beta) +
    // std::log(alpha + beta)) << "\n";

    if (prior == "laplace") {
      new_bound += arma::dot(
          gamma, arma::log(arma::abs(sigma)) + std::log(lambda) -
                     lambda * M_SQRT1_2 * M_2_SQRTPI * sigma %
                         arma::exp(-0.5 * arma::square(mu / sigma)) -
                     lambda * arma::erf(M_SQRT1_2 * mu / sigma) + 0.5);
    } else if (prior == "gaussian") {
      new_bound +=
          arma::accu(gamma % (arma::log(sigma) - std::log(lambda) -
                              0.5 * (arma::square(sigma) + arma::square(mu)) /
                                  std::pow(lambda, 2) +
                              0.5));

      // Rcpp::Rcout << "6: " << arma::accu(gamma % (arma::log(sigma) -
      // std::log(lambda) - 0.5 * (arma::square(sigma) + arma::square(mu)) /
      // std::pow(lambda, 2) + 0.5)) << "\n";
    }

    varbound.push_back(new_bound);

    // check for convergence
    arma::vec new_entr = entropy(gamma);
    if (arma::norm(new_entr - old_entr, "inf") <= tol) {
      break;
    } else {
      if (i > 5 && new_bound > old_bound) {
        break;
      }

      old_bound = new_bound;
      old_entr = new_entr;
    }
  }

  return Rcpp::List::create(
      Rcpp::Named("mu") = mu,
      Rcpp::Named("sigma") = sigma,
      Rcpp::Named("gamma") = gamma);
}
