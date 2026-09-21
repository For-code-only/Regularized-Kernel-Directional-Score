# Frozen main-ANM generator. The retained mechanisms, distributions,
# population normalization, RNG isolation and sample equations are identical
# to the successfully run prospective H13 source.

synthetic_function_names <- c(
  "asinh", "quintic", "sine_1.5", "tanh", "exponential", "softplus",
  "quadratic", "cubic_turning", "sine_pi", "multifrequency", "local_bump",
  "smooth_switch"
)

synthetic_noise_names <- c(
  "Gaussian", "Uniform", "Laplace", "t5", "Gamma_shape2_scale1",
  "lognormal_sdlog0.5", "lognormal_sdlog1", "Gaussian_contamination",
  "bimodal"
)

synthetic_cause_names <- c("Uniform", "Gaussian", "bimodal")

synthetic_mechanism <- function(x, function_id) {
  switch(as.integer(function_id),
    asinh(2 * x),
    x + 0.2 * x^5,
    sin(1.5 * x),
    tanh(3 * x),
    exp(1.5 * x),
    (pmax(4 * x, 0) + log1p(exp(-abs(4 * x)))) / 4,
    x^2,
    x^3 - 0.6 * x,
    sin(pi * x),
    sin(3 * pi * x) + 0.3 * sin(7 * pi * x),
    exp(-12 * (x - 0.25)^2),
    0.2 * x + (1.5 * x + 0.5) * plogis(6 * x),
    stop("Unknown function_id")
  )
}

synthetic_signs <- function(n) {
  2 * sample.int(2L, n, replace = TRUE) - 3
}

synthetic_draw_cause <- function(n, cause_id) {
  switch(as.integer(cause_id),
    runif(n, -1, 1),
    rnorm(n) / sqrt(3),
    (2 * synthetic_signs(n) + 0.5 * rnorm(n)) / sqrt(12.75),
    stop("Unknown cause_id")
  )
}

synthetic_draw_noise <- function(n, noise_id) {
  switch(as.integer(noise_id),
    rnorm(n),
    runif(n, -sqrt(3), sqrt(3)),
    synthetic_signs(n) * rexp(n, rate = sqrt(2)),
    sqrt(3 / 5) * rt(n, df = 5),
    (rgamma(n, shape = 2, scale = 1) - 2) / sqrt(2),
    {
      tau <- 0.5
      (exp(tau * rnorm(n)) - exp(tau^2 / 2)) /
        sqrt(expm1(tau^2) * exp(tau^2))
    },
    {
      tau <- 1
      (exp(tau * rnorm(n)) - exp(tau^2 / 2)) /
        sqrt(expm1(tau^2) * exp(tau^2))
    },
    {
      component_sd <- ifelse(runif(n) < 0.05, 5, 1)
      rnorm(n, sd = component_sd) / sqrt(2.2)
    },
    (2 * synthetic_signs(n) + 0.5 * rnorm(n)) / sqrt(4.25),
    stop("Unknown noise_id")
  )
}

synthetic_signal_moments <- function(function_id, cause_id, a, b) {
  if (length(function_id) != 1L || is.na(function_id) ||
      !(function_id %in% seq_len(12L)) ||
      length(cause_id) != 1L || is.na(cause_id) ||
      !(cause_id %in% seq_len(3L)) ||
      length(a) != 1L || !is.finite(a) || a < 0.8 || a > 1.2 ||
      length(b) != 1L || !is.finite(b) || b < -0.2 || b > 0.2)
    stop("Signal-moment arguments are outside the fixed main-ANM protocol")
  g <- function(x) synthetic_mechanism(a * x + b, function_id)
  quad <- function(f, lower, upper) {
    answer <- integrate(f, lower = lower, upper = upper,
                        subdivisions = 1000L, rel.tol = 1e-10,
                        abs.tol = 1e-11, stop.on.error = TRUE)
    if (!is.finite(answer$value)) stop("Nonfinite population integral")
    answer$value
  }
  expectation <- switch(as.integer(cause_id),
    function(h) quad(function(x) 0.5 * h(x), -1, 1),
    function(h) quad(function(z) h(z / sqrt(3)) * dnorm(z), -12, 12),
    function(h) {
      0.5 * quad(function(z) h((-2 + 0.5 * z) / sqrt(12.75)) *
                    dnorm(z), -12, 12) +
        0.5 * quad(function(z) h((2 + 0.5 * z) / sqrt(12.75)) *
                      dnorm(z), -12, 12)
    }
  )
  signal_mean <- expectation(g)
  signal_variance <- expectation(function(x) (g(x) - signal_mean)^2)
  if (!is.finite(signal_variance) || signal_variance <= 0)
    stop("Population signal variance is nonpositive or nonfinite")
  list(mean = signal_mean, sd = sqrt(signal_variance))
}

synthetic_with_seed <- function(seed, action) {
  if (length(seed) != 1L || !is.finite(seed) || seed < 1 ||
      seed > .Machine$integer.max || seed != floor(seed)) stop("Invalid seed")
  old_kind <- RNGkind()
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  old_seed <- if (had_seed) get(".Random.seed", envir = .GlobalEnv) else NULL
  on.exit({
    do.call(RNGkind, as.list(old_kind))
    if (had_seed) assign(".Random.seed", old_seed, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
      rm(".Random.seed", envir = .GlobalEnv)
  }, add = TRUE)
  set.seed(as.integer(seed), kind = "Mersenne-Twister", normal.kind = "Inversion",
           sample.kind = "Rejection")
  action()
}

synthetic_final_sample <- function(row) {
  if (!is.data.frame(row) || nrow(row) != 1L) stop("Supply one manifest row")
  required <- c("case_id", "n", "model", "function_id", "cause_id", "noise_id",
                "rho", "design_seed", "data_seed", "method_seed_base",
                "directional_accuracy_applicable")
  if (!all(required %in% names(row)) || anyNA(row[required]))
    stop("Incomplete main-ANM manifest row")
  if (row$model != "ANM" || !(row$function_id %in% seq_len(12L)) ||
      !(row$cause_id %in% seq_len(3L)) || !(row$noise_id %in% seq_len(9L)) ||
      !is.finite(row$rho) || row$rho <= 0 ||
      !isTRUE(row$directional_accuracy_applicable))
    stop("Manifest row is outside the frozen main-ANM protocol")
  design <- synthetic_with_seed(row$design_seed, function() list(
    a = runif(1L, .8, 1.2), b = runif(1L, -.2, .2),
    sign = synthetic_signs(1L)))
  moments <- synthetic_signal_moments(row$function_id, row$cause_id,
                                      design$a, design$b)
  sample <- synthetic_with_seed(row$data_seed, function() {
    cause <- synthetic_draw_cause(row$n, row$cause_id)
    epsilon <- synthetic_draw_noise(row$n, row$noise_id)
    signal <- design$sign *
      (synthetic_mechanism(design$a * cause + design$b, row$function_id) -
         moments$mean) / moments$sd
    list(cause = cause, effect = signal + row$rho * epsilon)
  })
  X <- cbind(cause = sample$cause, effect = sample$effect)
  if (any(!is.finite(X))) stop("Nonfinite generated sample: record failure, never redraw")
  list(X = X, truth = 1L, case_id = row$case_id,
       directional_accuracy_applicable = row$directional_accuracy_applicable,
       manifest = row, design = c(design, list(signal_mean = moments$mean,
                                               signal_sd = moments$sd)),
       generation_attempts = 1L)
}
