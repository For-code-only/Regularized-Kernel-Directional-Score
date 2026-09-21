# KCDC-D1, following Mitrovic, Sejdinovic and Teh (NeurIPS 2018),
# Section 3.1 and Equation (1). This is an independent base-R implementation.
# lambda is the normalized coefficient: the matrix ridge is n * lambda.
# The original paper does not specify a numerical lambda or delta default.

.kcdc_input <- function(z, standardize, name) {
  if (!is.numeric(z) || !is.null(dim(z)) || length(z) < 3L ||
      any(!is.finite(z))) {
    stop(name, " must be a finite numeric vector with at least three observations")
  }
  z <- as.numeric(z)
  if (standardize) {
    s <- stats::sd(z)
    if (!is.finite(s) || s <= 0) stop(name, " has no finite positive sample standard deviation")
    z <- (z - mean(z)) / s
  }
  z
}

.kcdc_kernel <- function(z, kernel, h) {
  kernel <- match.arg(kernel, c("rbf", "rq", "log"))
  if (length(h) != 1L || !is.finite(h) || h <= 0) {
    stop("Each kernel bandwidth must be finite and strictly positive")
  }
  d2 <- (outer(z, z, "-") / h)^2
  K <- switch(kernel,
              rbf = exp(-d2 / 2),
              rq = 1 / (1 + d2),
              log = -log1p(d2))
  if (any(!is.finite(K))) stop("Kernel evaluation produced nonfinite entries")
  K
}

kcdc_score <- function(x, y, lambda, kernel_in = "rbf", kernel_out = "rbf",
                       h_in = 1, h_out = 1, standardize = TRUE) {
  if (missing(lambda)) stop("lambda must be explicitly supplied; the paper gives no numerical default")
  if (length(lambda) != 1L || !is.finite(lambda) || lambda <= 0) {
    stop("lambda must be finite and strictly positive")
  }
  if (length(standardize) != 1L || is.na(standardize) || !is.logical(standardize)) {
    stop("standardize must be TRUE or FALSE")
  }
  if (length(x) != length(y)) stop("x and y must have the same number of observations")
  kernel_in <- match.arg(kernel_in, c("rbf", "rq", "log"))
  kernel_out <- match.arg(kernel_out, c("rbf", "rq", "log"))
  x <- .kcdc_input(x, standardize, "x")
  y <- .kcdc_input(y, standardize, "y")
  n <- length(x)
  ridge <- n * lambda
  if (!is.finite(ridge)) stop("n * lambda is not finite")

  # Neither Gram matrix is centered. Roles are fixed across both directions.
  K <- .kcdc_kernel(x, kernel_in, h_in)
  L <- .kcdc_kernel(y, kernel_out, h_out)
  system <- K
  diag(system) <- diag(system) + ridge
  if (kernel_in == "log") {
    # The literal log kernel in the paper is indefinite. Do not project it
    # to the PSD cone or add an unreported stabilizing penalty.
    A <- solve(system, K)
    solver <- "general_solve"
  } else {
    C <- chol(system)
    A <- backsolve(C, forwardsolve(t(C), K))
    solver <- "cholesky"
  }

  LA <- L %*% A
  squared_norm <- colSums(A * LA)
  # Only repair signs compatible with rounding in the diagonal dot products.
  roundoff <- 64 * .Machine$double.eps * n *
    pmax(colSums(abs(A) * abs(LA)), .Machine$double.xmin)
  if (any(!is.finite(squared_norm)) || any(squared_norm < -roundoff)) {
    stop("Conditional embedding squared norms are nonfinite or materially negative; the literal log response kernel need not be PSD")
  }
  corrected <- sum(squared_norm < 0)
  squared_norm[squared_norm < 0] <- 0
  norms <- sqrt(squared_norm)
  # Equation (1) uses the variance of norms, with divisor n, not var(norms).
  score <- mean((norms - mean(norms))^2)
  if (!is.finite(score)) stop("KCDC score is not finite")
  list(score = score, norms = norms, squared_norms = squared_norm,
       n = n, lambda = lambda, ridge = ridge,
       kernel_in = kernel_in, kernel_out = kernel_out,
       h_in = h_in, h_out = h_out, standardize = standardize,
       solver = solver, corrected_negative_norms = corrected)
}

kcdc_pair <- function(x, y, lambda, kernel_in = "rbf", kernel_out = "rbf",
                      h_in = 1, h_out = 1, standardize = TRUE,
                      delta = 0, relative_tolerance = 64 * .Machine$double.eps) {
  if (missing(lambda)) stop("lambda must be explicitly supplied; the paper gives no numerical default")
  if (length(delta) != 1L || !is.finite(delta) || delta < 0) {
    stop("delta must be finite and nonnegative")
  }
  if (length(relative_tolerance) != 1L || !is.finite(relative_tolerance) ||
      relative_tolerance < 0) {
    stop("relative_tolerance must be finite and nonnegative")
  }
  forward <- kcdc_score(x, y, lambda, kernel_in, kernel_out,
                        h_in, h_out, standardize)
  reverse <- kcdc_score(y, x, lambda, kernel_in, kernel_out,
                        h_in, h_out, standardize)
  sf <- forward$score
  sr <- reverse$score
  gap <- sr - sf  # positive favors X -> Y
  tolerance <- relative_tolerance * max(sf, sr)
  confidence <- if (min(sf, sr) > 0) {
    abs(gap) / min(sf, sr)
  } else if (max(sf, sr) > 0) {
    Inf
  } else {
    0
  }
  if (abs(gap) <= tolerance) {
    direction <- 0L
    status <- "numerical_tie"
  } else if (confidence < delta) {
    direction <- 0L
    status <- "confidence_abstention"
  } else {
    direction <- if (gap > 0) 1L else 2L
    status <- "directed"
  }
  # delta=0 means no substantive confidence rejection, not an original default.
  # This method has no RKDS/HSIC pre-screen. 1 = X -> Y, 2 = Y -> X, 0 = tie/abstention.
  list(direction = direction, status = status, score_forward = sf,
       score_reverse = sr, gap = gap, confidence = confidence,
       delta = delta, numerical_tolerance = tolerance,
       relative_tolerance = relative_tolerance,
       forward = forward, reverse = reverse)
}
