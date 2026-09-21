# R adapters retained by the paper experiment. RKDS is implemented in
# scientific_core.R; Python owns ANM-GP/ANM-KRR/PNL.

pair_standardize <- function(X) {
  X <- as.matrix(X)
  if (!is.numeric(X) || ncol(X) != 2L || nrow(X) < 4L || any(!is.finite(X)))
    stop("Expected a finite numeric n by 2 matrix, n >= 4")
  cbind(standardize(X[, 1L]), standardize(X[, 2L]))
}

compare_pair_scores <- function(sf, sr) {
  if (any(!is.finite(c(sf, sr)))) stop("Nonfinite direction score")
  tolerance <- 64 * .Machine$double.eps *
    max(abs(sf), abs(sr), .Machine$double.xmin)
  prediction <- if (abs(sr - sf) <= tolerance) 0L else if (sf < sr) 1L else 2L
  list(prediction = prediction,
       status = if (prediction == 0L) "numerical_tie" else "directed",
       score_forward = sf, score_reverse = sr)
}

reci_mon3_score <- function(x, y) {
  if (diff(range(x)) <= 0 || diff(range(y)) <= 0) stop("RECI requires nonconstant data")
  x <- (x - min(x)) / diff(range(x))
  y <- (y - min(y)) / diff(range(y))
  mean(lm.fit(cbind(1, x^3), y)$residuals^2)
}

reci_mon3_adapter <- function(X, seed) {
  Z <- pair_standardize(X)
  out <- compare_pair_scores(reci_mon3_score(Z[, 1L], Z[, 2L]),
                             reci_mon3_score(Z[, 2L], Z[, 1L]))
  out$method <- "RECI_MON3"
  out
}

make_kcdc_adapter <- function(configuration) {
  required <- c("lambda", "kernel_in", "kernel_out", "h_in", "h_out")
  if (!all(required %in% names(configuration))) stop("Incomplete KCDC configuration")
  if (configuration$kernel_in != "log" || configuration$kernel_out != "rq" ||
      configuration$h_in != 1 || configuration$h_out != 1)
    stop("Formal KCDC uses the paper's fixed log input and RQ output kernels")
  configuration <- as.list(configuration[required])
  force(configuration)
  function(X, seed) {
    out <- do.call(kcdc_pair, c(list(x = X[, 1L], y = X[, 2L]), configuration))
    out$prediction <- out$direction
    out$method <- "KCDC_D1"
    out
  }
}
