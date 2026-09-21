#!/usr/bin/env Rscript
# Prepare one or more frozen manifest rows and run only requested R methods.
# Python methods and PNL consume the canonical CSV written here.

args <- commandArgs(trailingOnly = TRUE)
file_arg <- commandArgs()[grepl("^--file=", commandArgs())][1]
code_dir <- normalizePath(dirname(sub("^--file=", "", file_arg)), mustWork = TRUE)
root <- normalizePath(file.path(code_dir, "..", ".."), mustWork = TRUE)
for (file in c("synthetic_final.R", "nonanm_cause.R", "nonanm_random.R",
               "preparation.R", "core.R", "kcdc.R", "kcdc_config.R", "adapters.R"))
  sys.source(file.path(code_dir, file), envir = .GlobalEnv)
source(file.path(code_dir, "scientific_core.R"))
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
core <- h13_load_core(file.path(code_dir, "core.R"))

with_sourcecpp_lock <- function(action, timeout = 900) {
  cache_root <- file.path(root, "cache")
  lock <- file.path(cache_root, ".sourcecpp.lock")
  dir.create(cache_root, recursive = TRUE, showWarnings = FALSE)
  started <- proc.time()[["elapsed"]]
  repeat {
    if (dir.create(lock, showWarnings = FALSE)) break
    age <- tryCatch(as.numeric(difftime(Sys.time(), file.info(lock)$mtime,
                                       units = "secs")), error = function(e) 0)
    if (is.finite(age) && age > timeout) {
      stale <- paste0(lock, ".stale.", Sys.getpid(), ".", as.integer(Sys.time()))
      if (file.rename(lock, stale)) unlink(stale, recursive = TRUE, force = TRUE)
    }
    if (proc.time()[["elapsed"]] - started > timeout)
      stop("Timed out waiting for the sourceCpp cache lock")
    Sys.sleep(0.05)
  }
  on.exit(unlink(lock, recursive = TRUE, force = TRUE), add = TRUE)
  action()
}

compile_core <- function() {
  if (!requireNamespace("Rcpp", quietly = TRUE)) stop("Rcpp is required")
  cache_dir <- file.path(root, "cache", "sourceCpp")
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  # sourceCpp reads and may update its cache index even with rebuild=FALSE.
  # Hold the process-shared lock through both cache access and dynlib loading.
  with_sourcecpp_lock(function()
    Rcpp::sourceCpp(file.path(code_dir, "permutation.cpp"), env = core$.accel,
                    cacheDir = cache_dir, rebuild = FALSE,
                    showOutput = FALSE, verbose = FALSE))
  core$.accel$fun <- core$.accel$extend1_permutation_sums
  stopifnot(is.function(core$.accel$fun))
  invisible(TRUE)
}

if (identical(args, "setup")) {
  compile_core()
  cat("R setup OK\n")
  quit(status = 0)
}
if (length(args) != 3L) stop("worker.R CASE.json OUTDIR all|core|pnl")
role <- args[3]
if (!role %in% c("all", "core", "pnl")) stop("Unknown worker role")
raw_jobs <- jsonlite::fromJSON(args[1], simplifyVector = FALSE)
jobs <- if (!is.null(raw_jobs$case_id)) list(raw_jobs) else raw_jobs
if (!is.list(jobs) || !length(jobs)) stop("CASE.json must contain at least one row")
outdir <- args[2]
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
if (role %in% c("all", "core")) compile_core()

atomic_json <- function(value, path) {
  temporary <- paste0(path, ".tmp.", Sys.getpid())
  jsonlite::write_json(value, temporary, auto_unbox = TRUE, digits = NA,
                       null = "null", na = "null")
  if (!file.rename(temporary, path)) stop("Atomic rename failed")
}

failure <- function(error) list(prediction = 0L, native_prediction = 0L,
  status = "implementation_failure", scoring_success = FALSE, success = FALSE,
  error = conditionMessage(error))

timed <- function(action) {
  started <- proc.time()[["elapsed"]]
  warnings <- character()
  result <- tryCatch(withCallingHandlers(action(), warning = function(warning) {
    warnings <<- c(warnings, conditionMessage(warning))
    invokeRestart("muffleWarning")
  }), error = failure)
  result$seconds <- proc.time()[["elapsed"]] - started
  result$warnings <- unique(warnings)
  result
}

csv_values <- function(value, default = character()) {
  if (is.null(value) || length(value) == 0L || is.na(value)) return(default)
  if (!nzchar(value)) return(character())
  values <- trimws(strsplit(as.character(value), ",", fixed = TRUE)[[1]])
  values[nzchar(values)]
}

for (i in seq_along(jobs)) {
  row <- jobs[[i]]
  case_id <- as.character(paper_scalar(row$case_id, "case_id"))
  csv_path <- file.path(outdir, paste0(i, ".csv"))
  json_path <- file.path(outdir, paste0(i, ".json"))
  prepared <- tryCatch(paper_prepare_case(row, csv_path, root), error = identity)
  if (inherits(prepared, "error")) {
    atomic_json(list(case_id = case_id, preparation_error = conditionMessage(prepared),
                     preparation_status = "failed"), json_path)
    next
  }
  Z <- prepared$data
  outputs <- list()
  progress <- function(incomplete = TRUE)
    atomic_json(list(case_id = case_id, metadata = prepared$metadata,
                     outputs = outputs, r_stage_incomplete = incomplete), json_path)
  progress()

  if (role %in% c("all", "core")) {
    suite <- if (is.null(row$suite)) as.character(row$split) else as.character(row$suite)
    is_sensitivity <- identical(suite, "sensitivity")
    defaults <- if (is_sensitivity) "RKDS" else c("RKDS", "RECI_MON3", "KCDC")
    requested <- unique(csv_values(row$requested_methods, defaults))
    allowed <- if (is_sensitivity) "RKDS" else c("RKDS", "RECI_MON3", "KCDC")
    if (any(!requested %in% allowed)) stop("Unexpected requested R method for suite")

    if ("RKDS" %in% requested && is_sensitivity) {
      full_grid <- h13_oat_configs()
      requested_configs <- unique(csv_values(row$requested_configurations,
                                             as.character(full_grid$config_id)))
      if (!length(requested_configs) || any(!requested_configs %in% full_grid$config_id))
        stop("Unknown requested RKDS sensitivity configuration")
      grid <- full_grid[match(requested_configs, full_grid$config_id), , drop = FALSE]
      started <- proc.time()[["elapsed"]]
      shared_screen <- tryCatch(h13_screen(Z, as.integer(row$screen_seed), core),
                                error = identity)
      results <- if (inherits(shared_screen, "error")) {
        setNames(lapply(requested_configs, function(name) {
          item <- failure(shared_screen)
          item$config_id <- name
          item$residual_mode <- "same_sample"
          item
        }), requested_configs)
      } else tryCatch(h13_score_grid(Z, seed = as.integer(row$screen_seed),
                                     grid = grid, core = core,
                                     screen = shared_screen), error = function(error)
        setNames(lapply(requested_configs, function(name) {
          item <- failure(error)
          item$config_id <- name
          item$residual_mode <- "same_sample"
          item
        }), requested_configs))
      shared_seconds <- proc.time()[["elapsed"]] - started
      timing_scope <- if (length(requested_configs) == 12L) "shared_12_config_grid" else
        paste0("shared_", length(requested_configs), "_config_retry_grid")
      for (configuration in requested_configs) {
        value <- results[[configuration]]
        value$method <- "RKDS"
        value$configuration <- configuration
        value$seconds <- shared_seconds
        value$timing_scope <- timing_scope
        value$warnings <- character()
        outputs[[paste0("RKDS_", configuration)]] <- value
        progress()
      }
    } else if ("RKDS" %in% requested) {
      outputs$RKDS <- timed(function()
        h13_scores(Z, seed = as.integer(row$screen_seed), core = core))
      outputs$RKDS$method <- "RKDS"
      outputs$RKDS$configuration <- "H13"
      progress()
    }

    if ("RECI_MON3" %in% requested) {
      outputs$RECI_MON3 <- timed(function()
        reci_mon3_adapter(Z, as.integer(row$reci_seed)))
      outputs$RECI_MON3$method <- "RECI_MON3"
      outputs$RECI_MON3$configuration <- "default"
      progress()
    }
    if ("KCDC" %in% requested) {
      adapter <- make_kcdc_adapter(kcdc_primary_configuration())
      outputs$KCDC <- timed(function() adapter(Z, as.integer(row$kcdc_seed)))
      # Norm vectors are not required for directional comparison.
      outputs$KCDC$forward <- NULL
      outputs$KCDC$reverse <- NULL
      outputs$KCDC$method <- "KCDC"
      outputs$KCDC$configuration <- "D1"
      progress()
    }
  }
  progress(FALSE)
  rm(Z, prepared, outputs)
  gc(verbose = FALSE)
}
