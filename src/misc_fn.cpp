#include "misc_fn.h"

arma::vec entropy(const arma::vec &x) {
  arma::vec ent(x.n_elem, arma::fill::zeros);
  for (arma::uword j = 0; j < x.n_elem; ++j) {
    // clamp values to avoid -Inf
    if ((x(j) > 1e-10) && (x(j) < 1 - 1e-10)) {
      ent(j) -= x(j) * std::log2(x(j)) + (1 - x(j)) * std::log2(1 - x(j));
    }
  }
  return ent;
}

double sigmoid(const double &x) {
  if (x > 32.0) {
    return 1;
  } else if (x < -32.0) {
    return 0;
  } else {
    return 1 / (1 + std::exp(-x));
  }
}

arma::vec gram_diag(const arma::mat &X) {
  arma::vec diag(X.n_cols);

  for (arma::uword i = 0; i < diag.n_elem; ++i) {
    diag(i) = std::pow(arma::norm(X.col(i)), 2);
  }
  return diag;
}
