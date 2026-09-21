# Canonical observation preparation shared by every method. The domain-specific
# generators and the frozen TCEP row index are the only permitted data sources.

paper_scalar <- function(x, name) {
  if (is.null(x) || length(x) != 1L || is.na(x)) stop("Missing scalar row field: ", name)
  x
}

paper_with_seed <- function(seed, action) {
  seed <- as.numeric(paper_scalar(seed, "seed"))
  if (!is.finite(seed) || seed < 0 || seed > .Machine$integer.max || seed != floor(seed))
    stop("Invalid R seed")
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

.paper_tcep_indices <- new.env(parent = emptyenv())
.paper_tcep_sources <- new.env(parent = emptyenv())

paper_load_tcep_indices <- function(path) {
  path <- normalizePath(path, mustWork = TRUE)
  if (exists(path, envir = .paper_tcep_indices, inherits = FALSE))
    return(get(path, envir = .paper_tcep_indices, inherits = FALSE))
  connection <- gzfile(path, open = "rt")
  on.exit(close(connection), add = TRUE)
  text <- paste(readLines(connection, warn = FALSE), collapse = "\n")
  document <- jsonlite::fromJSON(text, simplifyVector = FALSE)
  if (!identical(as.integer(document$schema_version), 1L) ||
      !identical(as.integer(document$index_base), 1L) ||
      !is.list(document$cases) || is.null(names(document$cases)) ||
      any(!nzchar(names(document$cases))) || anyDuplicated(names(document$cases)))
    stop("Invalid frozen TCEP index document")
  assign(path, document, envir = .paper_tcep_indices)
  document
}

paper_load_tcep_sources <- function(root) {
  path <- normalizePath(file.path(root, "data", "tcep", "pairs.csv"), mustWork = TRUE)
  if (exists(path, envir = .paper_tcep_sources, inherits = FALSE))
    return(get(path, envir = .paper_tcep_sources, inherits = FALSE))
  sources <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("pair_id", "cache_sha256", "n_raw", "n_finite")
  if (!all(required %in% names(sources)) || !nrow(sources))
    stop("Invalid frozen TCEP source inventory")
  pair_id <- suppressWarnings(as.integer(sources$pair_id))
  if (anyNA(pair_id) || any(pair_id < 1L) || anyDuplicated(pair_id) ||
      any(!grepl("^[0-9a-f]{64}$", sources$cache_sha256)))
    stop("Invalid frozen TCEP source inventory")
  rownames(sources) <- as.character(pair_id)
  assign(path, sources, envir = .paper_tcep_sources)
  sources
}

paper_read_tcep_source <- function(row, root) {
  pair_id <- as.numeric(paper_scalar(row$pair_id, "pair_id"))
  if (!is.finite(pair_id) || pair_id != floor(pair_id) || pair_id < 1L)
    stop("Invalid TCEP pair_id")
  pair_id <- as.integer(pair_id)
  expected_relative <- sprintf("data/tcep/cache/pair%04d.csv", pair_id)
  if (!identical(as.character(paper_scalar(row$data_file, "data_file")), expected_relative))
    stop("TCEP data_file does not match pair_id")
  sources <- paper_load_tcep_sources(root)
  source <- sources[as.character(pair_id), , drop = FALSE]
  if (nrow(source) != 1L) stop("TCEP pair is absent from frozen source inventory")
  expected <- as.character(source$cache_sha256[[1]])
  archive_path <- normalizePath(file.path(root, "data", "tcep", "observations.zip"), mustWork = TRUE)
  member <- sprintf("pair%04d.csv", pair_id)
  inventory <- utils::unzip(archive_path, list = TRUE)
  if (anyDuplicated(inventory$Name) || !setequal(inventory$Name, sprintf("pair%04d.csv", sources$pair_id)))
    stop("TCEP archive inventory mismatch")
  size <- inventory$Length[match(member, inventory$Name)]
  if (length(size) != 1L || is.na(size) || size <= 0 || size > .Machine$integer.max)
    stop("Invalid TCEP archive member size")
  connection <- unz(archive_path, member, open = "rb")
  on.exit(close(connection), add = TRUE)
  bytes <- readBin(connection, what = "raw", n = as.integer(size))
  if (length(bytes) != size) stop("Incomplete TCEP archive member")
  actual <- digest::digest(bytes, serialize = FALSE, algo = "sha256")
  if (!identical(actual, expected)) stop("TCEP raw cache SHA256 mismatch")
  list(data = as.matrix(read.csv(text = rawToChar(bytes), check.names = FALSE)), hash = actual)
}

paper_tcep_rows <- function(row, X, finite_rows, root) {
  index_path <- file.path(root, as.character(paper_scalar(row$index_file, "index_file")))
  document <- paper_load_tcep_indices(index_path)
  case_id <- as.character(paper_scalar(row$case_id, "case_id"))
  entry <- document$cases[[case_id]]
  if (is.null(entry) || !is.list(entry)) stop("TCEP case is absent from frozen indices: ", case_id)
  if (!identical(as.character(entry$case_id), case_id)) stop("TCEP index case_id mismatch")
  row_seed <- as.numeric(paper_scalar(row$row_seed, "row_seed"))
  entry_seed <- as.numeric(paper_scalar(entry$row_seed, "index row_seed"))
  if (!is.finite(row_seed) || row_seed != floor(row_seed) ||
      !is.finite(entry_seed) || entry_seed != row_seed)
    stop("TCEP frozen index row_seed mismatch")
  original_n <- as.numeric(paper_scalar(row$original_n, "original_n"))
  finite_n <- as.numeric(paper_scalar(row$finite_n, "finite_n"))
  if (!is.finite(original_n) || original_n != floor(original_n) || original_n != nrow(X) ||
      !is.finite(finite_n) || finite_n != floor(finite_n) || finite_n != length(finite_rows))
    stop("TCEP source row counts differ from manifest")
  selected <- unlist(entry$original_rows, use.names = FALSE)
  if (!length(selected) || anyNA(selected) || any(!is.finite(selected)) ||
      any(selected != floor(selected))) stop("TCEP original_rows must be nonempty integers")
  selected <- as.integer(selected)
  if (anyDuplicated(selected) || any(selected < 1L) || any(selected > nrow(X)) ||
      any(!selected %in% finite_rows)) stop("TCEP original_rows are invalid")
  expected_n <- as.integer(paper_scalar(row$n, "n"))
  if (length(selected) != expected_n) stop("TCEP frozen index length differs from manifest n")
  cap <- as.character(paper_scalar(row$cap, "cap"))
  if (identical(cap, "full")) {
    if (!identical(selected, as.integer(finite_rows)))
      stop("TCEP full index must contain every finite row in original order")
  } else {
    numeric_cap <- suppressWarnings(as.integer(cap))
    if (is.na(numeric_cap) || numeric_cap != 1000L ||
        length(selected) != min(length(finite_rows), numeric_cap))
      stop("TCEP index must implement exactly cap1000 or full")
  }
  reference_hash <- as.character(paper_scalar(entry$data_hash, "index data_hash"))
  if (!grepl("^[0-9a-f]{64}$", reference_hash)) stop("Invalid TCEP canonical data_hash")
  list(rows = selected, reference_hash = reference_hash)
}

paper_prepare_case <- function(row, path, root) {
  domain <- as.character(paper_scalar(row$domain, "domain"))
  if (!domain %in% c("anm", "tcep", "nonanm")) stop("Unknown experiment domain")
  numeric_names <- c("n", "function_id", "cause_id", "noise_id", "rho", "delta",
                     "replicate", "design_seed", "data_seed", "method_seed_base",
                     "presentation_seed", "row_seed", "generation_attempts",
                     "pair_seed_base", "library_seed")
  for (name in intersect(numeric_names, names(row))) {
    value <- row[[name]]
    if (length(value) == 1L && !is.na(value) &&
        !(is.character(value) && !nzchar(value))) row[[name]] <- as.numeric(value)
  }
  if ("directional_accuracy_applicable" %in% names(row) &&
      length(row$directional_accuracy_applicable) == 1L &&
      !is.na(row$directional_accuracy_applicable) &&
      nzchar(as.character(row$directional_accuracy_applicable)))
    row$directional_accuracy_applicable <- as.logical(row$directional_accuracy_applicable)

  design <- NULL
  generation_attempts <- 1L
  source_data_hash <- NULL
  if (identical(domain, "anm")) {
    generated <- synthetic_final_sample(as.data.frame(row, stringsAsFactors = FALSE))
    X <- generated$X
    truth <- generated$truth
    design <- generated$design
    generation_attempts <- generated$generation_attempts
  } else if (identical(domain, "nonanm")) {
    generated <- nonanm_random_sample(as.data.frame(row, stringsAsFactors = FALSE))
    X <- generated$X
    truth <- generated$truth
    design <- generated$design
    generation_attempts <- generated$generation_attempts
  } else {
    source <- paper_read_tcep_source(row, root)
    source_data_hash <- source$hash
    X <- source$data
    truth <- as.integer(paper_scalar(row$truth, "truth"))
  }
  X <- as.matrix(X)
  if (!is.numeric(X) || ncol(X) != 2L) stop("Prepared source must have two numeric columns")
  if (!truth %in% 1:2) stop("Truth must be encoded as 1 or 2")
  original_n <- nrow(X)
  finite_rows <- which(rowSums(!is.finite(X)) == 0L)
  if (identical(domain, "nonanm") && length(finite_rows) != original_n)
    stop("Nonfinite generated observations: no filtering or redraw")
  if (length(finite_rows) < 6L) stop("Fewer than six finite observations")

  reference_hash <- NULL
  if (identical(domain, "tcep")) {
    index <- paper_tcep_rows(row, X, finite_rows, root)
    selected <- index$rows
    reference_hash <- index$reference_hash
  } else {
    selected <- as.integer(finite_rows)
    if (length(selected) != as.integer(row$n))
      stop("Generated observation count differs from manifest n")
  }
  Z <- X[selected, , drop = FALSE]
  scale <- apply(Z, 2L, stats::sd)
  if (any(!is.finite(scale)) || any(scale <= 0)) stop("Constant/nonfinite input scale")
  Z <- sweep(sweep(Z, 2L, colMeans(Z), "-"), 2L, scale, "/")
  swapped <- paper_with_seed(row$presentation_seed,
                             function() sample(c(FALSE, TRUE), 1L))
  columns <- if (swapped) c(2L, 1L) else c(1L, 2L)
  Z <- Z[, columns, drop = FALSE]
  colnames(Z) <- c("X", "Y")
  write.csv(Z, path, row.names = FALSE)
  # All methods consume the exact bytes represented by this canonical CSV.
  Z <- as.matrix(read.csv(path, check.names = FALSE))
  data_hash <- digest::digest(file = path, algo = "sha256")

  metadata <- list(
    n_used = nrow(Z), original_n = original_n, finite_n = length(finite_rows),
    original_rows = as.integer(selected), finite_rows = as.integer(finite_rows),
    swapped = swapped, columns = columns, truth = as.integer(truth),
    presented_truth = if (swapped) 3L - truth else truth, design = design,
    data_hash = data_hash, generation_attempts = as.integer(generation_attempts))
  if (identical(domain, "tcep")) {
    metadata$source_data_hash <- source_data_hash
    metadata$tcep_reference_data_hash <- reference_hash
    metadata$tcep_reference_hash_match <- identical(data_hash, reference_hash)
    metadata$tcep_reference_hash_policy <- "audit_only_cross_runtime_csv_bytes"
    metadata$row_seed <- as.integer(paper_scalar(row$row_seed, "row_seed"))
    metadata$presentation_seed <- as.integer(paper_scalar(row$presentation_seed,
                                                          "presentation_seed"))
  }
  list(data = Z, metadata = metadata)
}
