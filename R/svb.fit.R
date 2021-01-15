#' Fit Approximate Posteriors to Sparse Linear and Logistic Models
#'
#' @description Main function of the \code{\link{sparsevb}} package. Computes
#'   mean-field posterior approximations for both linear and logistic regression
#'   models, including variable selection via sparsity-inducing spike and slab
#'   priors.
#'
#' @param X A numeric design matrix, each row of which represents a vector of
#'   covariates/independent variables/features. Though not required, it is
#'   recommended to center and scale the columns to have norm
#'   \code{sqrt(nrow(X))}.
#' @param Y An \code{nrow(X)}-dimensional response vector, numeric if
#'   \code{family = "linear"} and binary if \code{family = "logistic"}.
#' @param family A character string selecting the regression model, either
#'   \code{"linear"} or \code{"logistic"}.
#' @param slab A character string specifying the prior slab density, either
#'   \code{"laplace"} or \code{"gaussian"}.
#' @param mu An \code{ncol(X)}-dimensional numeric vector, serving as initial
#'   guess for the variational means. If omitted, \code{mu} will be estimated
#'   via ridge regression to initialize the coordinate ascent algorithm.
#' @param sigma A positive \code{ncol(X)}-dimensional numeric vector, serving as
#'   initial guess for the variational standard deviations.
#' @param gamma An \code{ncol(X)}-dimensional vector of probabilities, serving
#'   as initial guess for the variational inclusion probabilities. If omitted,
#'   \code{gamma} will be estimated via LASSO regression to initialize the
#'   coordinate ascent algorithm.
#' @param alpha A positive numeric value, parametrizing the beta hyper-prior on
#'   the inclusion probabilities. If omitted, \code{alpha} will be chosen
#'   empirically via LASSO regression.
#' @param beta A positive numeric value, parametrizing the beta hyper-prior on
#'   the inclusion probabilities. If omitted, \code{beta} will be chosen
#'   empirically via LASSO regression.
#' @param prior_scale A numeric value, controlling the scale parameter of the
#'   prior slab density. Used as the scale parameter \eqn{\lambda} when
#'   \code{prior = "laplace"}, or as the standard deviation \eqn{\sigma} if
#'   \code{prior = "gaussian"}.
#' @param update_order A permutation of \code{1:ncol(X)}, giving the update
#'   order of the coordinate-ascent algorithm. If omitted, a data driven
#'   updating order is used, see \emph{Ray and Szabo (2020)} in \emph{Journal of
#'   the American Statistical Association} for details.
#' @param intercept A Boolean variable, controlling if an intercept should be
#'   included. NB: This feature is still experimental in logistic regression.
#' @param noise_sd A positive numerical value, serving as estimate for the
#'   residual noise standard deviation in linear regression. If missing it will
#'   be estimated, see \code{estimateSigma} from the \code{selectiveInference}
#'   package for more details. Has no effect when \code{family = "logistic"}.
#' @param max_iter A positive integer, controlling the maximum number of
#'   iterations for the variational update loop.
#' @param tol A small, positive numerical value, controlling the termination
#'   criterion for maximum absolute differences between binary entropies of
#'   successive iterates.
#'
#' @return The approximate mean-field posterior, given as a named list
#'   containing numeric vectors \code{"mu"}, \code{"sigma"}, \code{"gamma"}, and
#'   a value \code{"intercept"}. The latter is set to \code{NA} in case
#'   \code{intercept = FALSE}. In mathematical terms, the conditional
#'   distribution of each \eqn{\theta_j} is given by \deqn{\theta_j\mid \mu_j,
#'   \sigma_j, \gamma_j \sim_{ind.} \gamma_j N(\mu_j, \sigma^2) + (1-\gamma_j)
#'   \delta_0.}
#'
#' @examples
#' 
#' ### Simulate a linear regression problem of size n times p, with sparsity level s ###
#'
#' n <- 250
#' p <- 500
#' s <- 5
#'
#' ### Generate toy data ###
#'
#' X <- matrix(rnorm(n*p), n, p) #standard Gaussian design matrix
#'
#' theta <- numeric(p)
#' theta[sample.int(p, s)] <- runif(s, -3, 3) #sample non-zero coefficients in random locations
#'
#' pos_TR <- as.numeric(theta != 0) #true positives
#'
#' Y <- X %*% theta + rnorm(n) #add standard Gaussian noise
#'
#' ### Run the algorithm in linear mode with Laplace prior and prioritized initialization ###
#'
#' test <- svb.fit(X, Y, family = "linear")
#'
#' posterior_mean <- test$mu * test$gamma #approximate posterior mean
#'
#' pos <- as.numeric(test$gamma > 0.5) #significant coefficients
#'
#' ### Assess the quality of the posterior estimates ###
#'
#' TPR <- sum(pos[which(pos_TR == 1)])/sum(pos_TR) #True positive rate
#'
#' FDR <- sum(pos[which(pos_TR != 1)])/max(sum(pos), 1) #False discovery rate
#'
#' L2 <- sqrt(sum((posterior_mean - theta)^2)) #L_2-error
#'
#' MSPE <- sqrt(sum((X %*% posterior_mean - Y)^2)/n) #Mean squared prediction error
#'
#' @details Suppose \eqn{\theta} is the \eqn{p}-dimensional true parameter. The
#'   spike-and-slab prior for \eqn{\theta} may be represented by the
#'   hierarchical scheme \deqn{w \sim \mathrm{Beta}(\alpha, \beta),} \deqn{z_j
#'   \mid w \sim_{i.i.d.} \mathrm{Bernoulli}(w),} \deqn{\theta_j\mid z_j
#'   \sim_{ind.} (1-z_j)\delta_0 + z_j g.} Here, \eqn{\delta_0} represents the
#'   Dirac measure at \eqn{0}. The slab \eqn{g} may be taken either as a
#'   \eqn{\mathrm{Laplace}(0,\lambda)} or \eqn{N(0,\sigma^2)} density. The
#'   former has centered density \deqn{f_\lambda(x) = \frac{\lambda}{2}
#'   e^{-\lambda |x|}.} Given \eqn{\alpha} and \eqn{\beta}, the beta hyper-prior
#'   has density \deqn{b(x\mid \alpha, \beta) = \frac{x^{\alpha - 1}(1 -
#'   x)^{\beta - 1}}{\int_0^1 t^{\alpha - 1}(1 - t)^{\beta - 1}\ \mathrm{d}t}.}
#'   A straightforward integration shows that the prior inclusion probability of
#'   a coefficient is \eqn{\frac{\alpha}{\alpha + \beta}}.
#'
#' @export
svb.fit <- function(X,
                    Y,
                    family = c("linear", "logistic"),
                    slab = c("laplace", "gaussian"),
                    mu,
                    sigma = rep(1, ncol(X)),
                    gamma,
                    alpha,
                    beta,
                    prior_scale = 1,
                    update_order,
                    intercept = FALSE,
                    noise_sd,
                    max_iter = 1000,
                    tol = 1e-5) {
    
    #extract problem dimensions
    n = nrow(X)
    p = ncol(X)
    
    #rescale data if necessary
    if(match.arg(family) == "linear" && missing(noise_sd)) {
        noise_sd = estimateSigma(X, Y)$sigmahat
    } else if(match.arg(family) == "logistic") {
        noise_sd = 1 
    }
    X = X/noise_sd
    Y = Y/noise_sd
    
    #compute initial estimator for mu
    if(missing(mu)) {
        cvfit = cv.glmnet(X, Y, family = ifelse(match.arg(family) == "linear", "gaussian", "binomial"), intercept = intercept, alpha = 0)
        mu = as.numeric(coef(cvfit, s = "lambda.min"))
        
        if(intercept) {
            mu = c(mu[2:(p+1)], mu[1])
        } else {
            mu = mu[2:(p+1)]
        }
    } else if (intercept) {
        mu = c(mu, 0)
    }
    
    #generate prioritized updating order
    if(missing(update_order)) {
        update_order = order(abs(mu[1:p]), decreasing = TRUE)
        update_order = update_order - 1
    }
    
    #compute initial estimators for alpha, beta, and gamma
    if(missing(gamma) || missing(alpha) || missing(beta)) {
        cvfit = cv.glmnet(X, Y, family = ifelse(match.arg(family) == "linear", "gaussian", "binomial"), intercept = intercept, alpha = 1)
        
        s_hat = length(which(coef(cvfit, s = "lambda.1se")[-1] != 0))
        
        if(missing(alpha)) {
            alpha = s_hat
        }
        if(missing(beta)) {
            beta = p - s_hat
        }
        if(missing(gamma)){
            gamma = rep(s_hat/p, p)
            gamma[which(coef(cvfit, s = "lambda.1se")[-1] != 0)] = 1
        }
    }
    
    #add intercept
    if(intercept){
        sigma = c(sigma, 1)
        gamma = c(gamma, 1)
        update_order = c(p, update_order)
        X = cbind(X, rep(1/noise_sd, n))
    }
    
    #match internal function call and generate list of arguments
    fn = paste("fit", match.arg(family), sep = '_')
    arg = list(X, Y, mu, sigma, gamma, alpha, beta, prior_scale, update_order, match.arg(slab), max_iter, tol)
    
    #perform chosen computation
    approximate_posterior = lapply(do.call(fn, arg), as.numeric)
    
    #convert results to R-style vectors since RcppArmadillo returns in matrix form
    return(list(mu = approximate_posterior$mu[1:p], sigma = approximate_posterior$sigma[1:p], gamma = approximate_posterior$gamma[1:p], intercept = ifelse(intercept, mu[p+1], NA)))
}
