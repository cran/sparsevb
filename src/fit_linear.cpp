#include "fit_linear.h"

Rcpp::List fit_linear(const arma::mat &X, const arma::vec &Y, arma::vec mu,
                      arma::vec sigma, arma::vec gamma, const double &alpha,
                      const double &beta, const double &lambda,
                      const arma::uvec &update_order, const std::string &prior,
                      const size_t &max_iter, const double &tol) {
  // initalize entropy loss function
  arma::vec old_entr = entropy(gamma);

  // pre-process update parameters
  arma::rowvec YX_vec = Y.t() * X;
  arma::vec half_diag = 0.5 * gram_diag(X);
  arma::vec approx_mean = gamma % mu;
  arma::vec X_appm = X * approx_mean;

  double const_lodds = std::log(alpha) - std::log(beta) + 0.5;
  if (prior == "laplace") {
    const_lodds += 0.5 * std::log(M_PI) + std::log(lambda) - 0.5 * M_LN2;

  } else if (prior == "gaussian") {
    const_lodds -= std::log(lambda);

    // implements equation (8) of Carbonetto et al.
    sigma = 1 / arma::sqrt(2 * half_diag + 1 / std::pow(lambda, 2));
  }

  // iteration loop
  for (size_t i = 0; i < max_iter; ++i) {
    // coordinate update loop
    for (arma::uword k = 0; k < mu.n_elem; ++k) {
      // check if interrupt signal was sent from R
      Rcpp::checkUserInterrupt();

      // the current update dimension
      arma::uword j = update_order(k);

      // delete the j-th column from X * approx_mean
      X_appm -= approx_mean(j) * X.col(j);

      if (prior == "laplace") {
        // initialize L-BFGS optimizer from ensmallen
        ens::L_BFGS optim;
        
        // start optimization at previous value
        arma::mat x(2, 1);
        x(0) = mu(j);
        x(1) = sigma(j);

        // implements equation (16) of the paper
        laplace_obj_fn f(half_diag(j), arma::dot(X.col(j), X_appm) - YX_vec(j),
                         lambda);

        // optimize and save function value
        double opt = optim.Optimize(f, x);
        mu(j) = x(0);
        // sigma(j) = std::abs(x(1));
        sigma(j) = x(1);

        // implements equation (17) of the paper
        gamma(j) = sigmoid(const_lodds - opt);

      } else if (prior == "gaussian") {
        // implements equation (9) of Carbonetto et al.
        mu(j) =
            std::pow(sigma(j), 2) * (YX_vec(j) - arma::dot(X.col(j), X_appm));

        // implements equation (10) of Carbonetto et al.
        gamma(j) = sigmoid(const_lodds + std::log(sigma(j)) +
                           0.5 * std::pow(mu(j) / sigma(j), 2));
      }

      // add j-th column with updated values
      approx_mean(j) = gamma(j) * mu(j);
      X_appm += approx_mean(j) * X.col(j);
    }

    // check for convergence
    arma::vec new_entr = entropy(gamma);
    if (arma::norm(new_entr - old_entr, "inf") <= tol) {
      break;
    } else {
      old_entr = new_entr;
    }
  }

  return Rcpp::List::create(Rcpp::Named("mu") = mu,
                            Rcpp::Named("sigma") = sigma,
                            Rcpp::Named("gamma") = gamma);
}
