# Randomized simple mechanisms in paired worlds; no generation-time screen.
# Source synthetic_final.R and nonanm_cause.R first. Frozen mechanisms,
# unit-variance distribution formulas and isolated RNG are reused unchanged;
# nonanm_signal_moments normalizes against the newly selected cause law.

nonanm_random_models <- c("ANM_control", "heteroscedastic", "post_nonlinear",
                          "multiplicative")
nonanm_random_function_ids <- c(7L, 8L, 6L, 4L, 10L, 11L)
nonanm_random_noise_ids <- c(1L, 3L, 5L, 9L)
nonanm_random_cause_ids <- nonanm_random_noise_ids
nonanm_random_outer_ids <- c("cubic", "asinh")

nonanm_random_outer <- function(z, outer_id, delta = .2) {
  if (length(delta) != 1L || !is.finite(delta) || delta <= 0)
    stop("Outer delta must be positive and finite")
  switch(as.character(outer_id),
    cubic = z + delta * z^3,
    asinh = asinh(delta * z) / delta,
    stop("Unknown outer_id"))
}

nonanm_random_inverse_outer <- function(y, outer_id, delta = .2) {
  if (length(delta) != 1L || !is.finite(delta) || delta <= 0)
    stop("Outer delta must be positive and finite")
  switch(as.character(outer_id),
    cubic = 2 / sqrt(3 * delta) *
      sinh(asinh(3 * sqrt(3 * delta) * y / 2) / 3),
    asinh = sinh(delta * y) / delta,
    stop("Unknown outer_id"))
}

nonanm_random_validate_row <- function(row) {
  if (!is.data.frame(row) || nrow(row) != 1L) stop("Supply one manifest row")
  required <- c("case_id", "world_id", "split", "domain", "n", "model", "function_id", "cause_id",
                "noise_id", "rho", "delta", "outer_id", "design_seed", "data_seed",
                "method_seed_base", "pair_seed_base", "library_seed",
                "generation_attempts", "directional_accuracy_applicable")
  if (!all(required %in% names(row))) stop("Incomplete randomized-world manifest row")
  if (anyNA(row[required])) stop("Missing randomized-world manifest value")
  is_smoke <- identical(as.character(row$split), "smoke")
  valid_size <- if (is_smoke) row$n == 80L else row$n %in% c(200L, 500L, 1000L)
  if (!valid_size || row$domain != "nonanm" ||
      !(row$split %in% c("nonanm", "smoke")) ||
      !(row$model %in% nonanm_random_models) ||
      !(row$function_id %in% nonanm_random_function_ids) ||
      !(row$cause_id %in% nonanm_random_cause_ids) ||
      !(row$noise_id %in% nonanm_random_noise_ids) ||
      row$rho != 1 || row$delta != .2 ||
      !(row$outer_id %in% nonanm_random_outer_ids) ||
      row$generation_attempts != 1L || !isTRUE(row$directional_accuracy_applicable))
    stop("Manifest row is outside the frozen randomized-world protocol")
  invisible(TRUE)
}

nonanm_random_sample <- function(row) {
  nonanm_random_validate_row(row)
  if (!exists("nonanm_signal_moments", mode = "function"))
    stop("Source src/r/nonanm_cause.R before sampling the v2 random-cause design")
  design <- synthetic_with_seed(row$design_seed, function() list(
    a = runif(1L, .8, 1.2), b = runif(1L, -.2, .2),
    sign = synthetic_signs(1L)))
  moments <- nonanm_signal_moments(row$function_id, row$cause_id, design$a, design$b)
  # All four structures draw exactly the same (X, epsilon) in the same order.
  # Independent n-specific worlds have different data and design seeds.
  draws <- synthetic_with_seed(row$data_seed, function() list(
    cause = synthetic_draw_noise(row$n, row$cause_id),
    epsilon = synthetic_draw_noise(row$n, row$noise_id)))
  signal <- design$sign *
    (synthetic_mechanism(design$a * draws$cause + design$b, row$function_id) -
       moments$mean) / moments$sd
  latent <- signal + row$rho * draws$epsilon
  # All four cause laws have E[X]=0, Var(X)=1, so E[h(X)^2]=1.
  # h is a SIGNED noise coefficient, not a positive conditional standard
  # deviation: it may be negative when X < -1/delta. No clipping, abs or
  # remapping is applied; conditional variance is rho^2*h(X)^2.
  noise_coefficient <- (1 + row$delta * draws$cause) / sqrt(1 + row$delta^2)
  effect <- switch(as.character(row$model),
    ANM_control = latent,
    heteroscedastic = signal + row$rho * noise_coefficient * draws$epsilon,
    post_nonlinear = nonanm_random_outer(latent, row$outer_id, row$delta),
    # 1 + delta Y = exp(delta g(X)) * exp(delta rho epsilon).
    # expm1 avoids cancellation while retaining the multiplicative structure.
    multiplicative = expm1(row$delta * latent) / row$delta,
    stop("Unknown model"))
  X <- cbind(cause = draws$cause, effect = effect)
  if (any(!is.finite(X))) stop("Nonfinite generated sample: record failure, never redraw")
  list(X = X, truth = 1L, case_id = row$case_id,
       directional_accuracy_applicable = row$directional_accuracy_applicable,
       manifest = row,
       design = c(design, list(signal_mean = moments$mean, signal_sd = moments$sd)),
       generation_attempts = 1L,
       # Retained for generation QA only; algorithms receive X alone.
       epsilon = draws$epsilon, signal = signal, latent = latent)
}
