# RKDS paper-experiment scientific engine. All same-sample equations and the
# marginal screen are delegated to the frozen core.R. No truth label is
# accepted by any scoring function.

h13_load_core <- function(core_path = file.path("src", "r", "core.R")) {
  if (!file.exists(core_path)) stop("Missing frozen RKDS core: ", core_path)
  core <- new.env(parent = globalenv())
  sys.source(core_path, envir = core)
  required <- c("center_gram", "gaussian_gram", "screen_hsic", "residual_operator",
                "residual_energies", "normalized_score")
  if (!all(vapply(required, function(x) is.function(core[[x]]), logical(1))))
    stop("Frozen RKDS source is incomplete")
  core
}

h13_config <- function() {
  list(config_id = "H13", theta = 1e-6, sigma1 = .2, sigma2 = .2,
       sigma3 = 1.6, screen_sigma = .4, hsic_B = 999L, hsic_alpha = .05,
       tie_tol = 1e-12)
}

h13_oat_configs <- function() {
  base <- h13_config()
  out <- list(base)
  values <- list(theta = c(1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3),
                 sigma1 = c(.1, .2, .4), sigma2 = c(.1, .2, .4),
                 sigma3 = c(.8, 1.6, 3.2))
  for (parameter in names(values)) for (value in values[[parameter]]) {
    if (value == base[[parameter]]) next
    cfg <- base
    cfg[[parameter]] <- value
    cfg$config_id <- paste0("H13_", parameter, "_", format(value, scientific = TRUE,
                                                          trim = TRUE))
    out[[length(out) + 1L]] <- cfg
  }
  out <- do.call(rbind, lapply(out, as.data.frame))
  rownames(out) <- NULL
  stopifnot(nrow(out) == 12L, !anyDuplicated(out[c("theta", "sigma1", "sigma2", "sigma3")]))
  out
}

h13_validate_config <- function(cfg, core) {
  cfg <- utils::modifyList(h13_config(), as.list(cfg))
  fields <- c("theta", "sigma1", "sigma2", "sigma3")
  if (!all(vapply(cfg[fields], function(x) length(x) == 1L && is.finite(x) && x > 0,
                  logical(1)))) stop("Invalid H13 parameters")
  if (!identical(as.numeric(cfg$screen_sigma), as.numeric(core$SCREEN_SIGMA)) ||
      !identical(as.numeric(cfg$hsic_alpha), as.numeric(core$SCREEN_ALPHA)) ||
      !identical(as.integer(cfg$hsic_B), 999L) ||
      !identical(as.numeric(cfg$tie_tol), 1e-12))
    stop("H13 screen/tie protocol must remain frozen")
  cfg
}

h13_validate_data <- function(Z) {
  if (!is.matrix(Z) || ncol(Z) != 2L || nrow(Z) < 5L || any(!is.finite(Z)))
    stop("Z must be the finite, already standardized n by 2 presented data")
  if (any(apply(Z, 2L, stats::sd) == 0)) stop("Constant data column")
  invisible(TRUE)
}

h13_preserve_rng <- function(fun) {
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit(if (had_seed) assign(".Random.seed", old_seed, envir = .GlobalEnv) else
    if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
      rm(".Random.seed", envir = .GlobalEnv), add = TRUE)
  fun()
}

h13_screen <- function(Z, seed, core) {
  h13_preserve_rng(function() core$screen_hsic(Z[, 1L], Z[, 2L], seed = seed, B = 999L))
}

h13_context <- function(Z, core) {
  context <- new.env(parent = emptyenv())
  context$Z <- Z
  context$core <- core
  context$grams <- new.env(parent = emptyenv())
  context$eigenvalues <- new.env(parent = emptyenv())
  # Retain just the current ridge pair, not every theta's n by n matrices.
  context$operator_key <- NULL
  context$operators <- NULL
  context$energies <- new.env(parent = emptyenv())
  context
}

h13_grams <- function(context, sigma) {
  key <- sprintf("%.17g", sigma)
  if (!exists(key, envir = context$grams, inherits = FALSE))
    assign(key, lapply(1:2, function(j) context$core$center_gram(
      context$core$gaussian_gram(context$Z[, j], sigma))), envir = context$grams)
  get(key, envir = context$grams, inherits = FALSE)
}

h13_eigenvalues <- function(Kc) {
  # The exact spectral shift gives eigenvalues of A=Kc/n+theta*I. Centered
  # Kc itself is singular and is deliberately NOT the reported condition object.
  eigen((Kc + t(Kc)) / (2 * nrow(Kc)), symmetric = TRUE, only.values = TRUE)$values
}

h13_spectrum <- function(values, theta) {
  shifted <- values + theta
  lo <- min(shifted)
  hi <- max(shifted)
  c(lambda_min = lo, lambda_max = hi,
    condition_number = if (lo > 0) hi / lo else Inf)
}

h13_same_sample <- function(context, cfg, diagnostics) {
  core <- context$core
  measurement <- h13_grams(context, cfg$sigma1)
  smoothing <- h13_grams(context, cfg$sigma2)
  operator_key <- sprintf("%.17g/%.17g", cfg$sigma2, cfg$theta)
  if (!identical(context$operator_key, operator_key)) {
    context$operators <- lapply(smoothing, core$residual_operator, theta = cfg$theta)
    context$operator_key <- operator_key
    context$energies <- new.env(parent = emptyenv())
  }
  energy_key <- sprintf("%.17g", cfg$sigma1)
  if (!exists(energy_key, envir = context$energies, inherits = FALSE))
    assign(energy_key, lapply(1:2, function(j)
      core$residual_energies(context$operators[[j]], measurement[[3L - j]])),
      envir = context$energies)
  energy <- get(energy_key, envir = context$energies, inherits = FALSE)
  spectrum <- replicate(2L, c(lambda_min = NA_real_, lambda_max = NA_real_,
                             condition_number = NA_real_), simplify = FALSE)
  if (diagnostics) {
    eigen_key <- sprintf("%.17g", cfg$sigma2)
    if (!exists(eigen_key, envir = context$eigenvalues, inherits = FALSE))
      assign(eigen_key, lapply(smoothing, h13_eigenvalues), envir = context$eigenvalues)
    spectrum <- lapply(get(eigen_key, envir = context$eigenvalues, inherits = FALSE),
                       h13_spectrum, theta = cfg$theta)
  }
  list(energy = energy, spectrum = spectrum)
}

h13_scores <- function(Z, seed, cfg = h13_config(), core, screen = NULL,
                       diagnostics = TRUE, return_energies = FALSE,
                       .context = NULL) {
  h13_validate_data(Z)
  cfg <- h13_validate_config(cfg, core)
  if (is.null(screen)) screen <- h13_screen(Z, seed, core)
  if (!is.list(screen) || length(screen$p) != 1L || !is.finite(screen$p) ||
      screen$p < 0 || screen$p > 1) stop("Invalid marginal screen result")
  # Always compute both directional scores, including screen non-rejection.
  if (is.null(.context)) .context <- h13_context(Z, core)
  measurement <- h13_grams(.context, cfg$sigma1)
  fit <- h13_same_sample(.context, cfg, diagnostics)
  scores <- lapply(1:2, function(j)
    core$normalized_score(measurement[[j]], fit$energy[[j]]$d, cfg$sigma3))
  q <- vapply(scores, function(s) unname(s["q"]), numeric(1))
  delta <- q[2L] - q[1L]
  success <- all(is.finite(q))
  tie <- success && abs(delta) <= cfg$tie_tol
  direction <- if (!success || tie) 0L else if (delta > 0) 1L else 2L
  screen_passed <- screen$p <= cfg$hsic_alpha
  native <- if (success && screen_passed) direction else 0L
  status <- if (!success) "zero_denominator" else if (!screen_passed)
    "screen_nonrejection" else if (tie) "numerical_tie" else "directed"
  out <- list(config_id = cfg$config_id, residual_mode = "same_sample",
    prediction = native, native_prediction = native,
    status = status, scoring_success = success, score_forward = q[1L],
    score_reverse = q[2L], delta = delta, confidence = abs(delta),
    exact_tie = success && q[1L] == q[2L], numerical_tie = tie,
    screen_p = screen$p, screen_hsic = screen$hsic, screen_rejected = screen_passed,
    screen_B = 999L, theta = cfg$theta, sigma1 = cfg$sigma1,
    sigma2 = cfg$sigma2, sigma3 = cfg$sigma3,
    lambda_n = nrow(Z) * cfg$theta,
    spectral_object = "centered_input_Gram/train_n + theta*I",
    spectral_summary = "actual_system",
    energy_truncations = sum(vapply(fit$energy, `[[`, numeric(1), "truncated")))
  for (j in 1:2) {
    suffix <- c("forward", "reverse")[j]
    for (field in c("hsic", "denominator", "input_self_hsic", "energy_self_hsic"))
      out[[paste0(field, "_", suffix)]] <- unname(scores[[j]][field])
    for (field in names(fit$spectrum[[j]]))
      out[[paste0(field, "_", suffix)]] <- unname(fit$spectrum[[j]][field])
    out[[paste0("minimum_raw_energy_", suffix)]] <- fit$energy[[j]]$minimum
    out[[paste0("energy_sd_", suffix)]] <- stats::sd(fit$energy[[j]]$d)
    if (return_energies) out[[paste0("energies_", suffix)]] <- fit$energy[[j]]$d
  }
  out
}

h13_score_grid <- function(Z, seed, grid = h13_oat_configs(), core, screen = NULL,
                           diagnostics = TRUE, return_energies = FALSE) {
  h13_validate_data(Z)
  if (!is.data.frame(grid) || !nrow(grid) || anyDuplicated(grid$config_id))
    stop("Grid must have unique config_id rows")
  if (is.null(screen)) screen <- h13_screen(Z, seed, core)
  context <- h13_context(Z, core)
  out <- vector("list", nrow(grid))
  # Reusing a current (sigma2,theta) operator is important for sigma1/sigma3 OAT.
  execution_order <- order(grid$sigma2, grid$theta, grid$sigma1, grid$sigma3)
  for (i in execution_order) {
    cfg <- as.list(grid[i, , drop = FALSE])
    out[[i]] <- tryCatch(h13_scores(Z, seed, cfg, core, screen,
                           diagnostics, return_energies = return_energies,
                           .context = context), error = function(e) {
      # A failed sensitivity configuration does not erase other paired results.
      failure <- list(config_id = cfg$config_id, residual_mode = "same_sample",
        prediction = 0L, native_prediction = 0L,
        status = "implementation_failure", scoring_success = FALSE,
        score_forward = NA_real_, score_reverse = NA_real_, delta = NA_real_,
        confidence = NA_real_, exact_tie = FALSE, numerical_tie = FALSE,
        screen_p = screen$p, screen_hsic = screen$hsic,
        screen_rejected = screen$p <= .05, screen_B = 999L,
        theta = cfg$theta, sigma1 = cfg$sigma1, sigma2 = cfg$sigma2,
        sigma3 = cfg$sigma3, lambda_n = nrow(Z) * cfg$theta,
        spectral_object = "centered_input_Gram/train_n + theta*I",
        spectral_summary = "actual_system", energy_truncations = NA_integer_,
        error = conditionMessage(e))
      for (suffix in c("forward", "reverse")) for (field in c("hsic",
          "denominator", "input_self_hsic", "energy_self_hsic", "lambda_min",
          "lambda_max", "condition_number", "minimum_raw_energy", "energy_sd"))
        failure[[paste0(field, "_", suffix)]] <- NA_real_
      failure
    })
  }
  names(out) <- as.character(grid$config_id)
  out
}
