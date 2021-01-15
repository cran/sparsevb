#' @details For details as they pertain to using the package, consult the
#'   \code{\link{svb.fit}} function help page. Detailed descriptions and
#'   derivations of the variational algorithms with Laplace slabs may be found
#'   in the references.
#'
#' @references \itemize{ \item Ray K. and Szabo B. Variational Bayes for
#'   high-dimensional linear regression with sparse priors. (2020).
#'   \emph{Journal of the American Statistical Association}. \item Ray K., Szabo
#'   B., and Clara G. Spike and slab variational Bayes for high dimensional
#'   logistic regression. (2020). \emph{Advances in Neural Information
#'   Processing Systems 33}.}
#'
#'
#'
#' @importFrom Rcpp evalCpp
#' @importFrom selectiveInference estimateSigma
#' @importFrom glmnet cv.glmnet
#' @importFrom stats coef
#' @useDynLib sparsevb, .registration = TRUE
"_PACKAGE"

NULL