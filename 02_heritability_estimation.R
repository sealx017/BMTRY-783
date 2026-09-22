###############################################################################
# SNP-HERITABILITY ESTIMATION
#
# Demonstrates:
#   1. Construction of a genomic relationship matrix (GRM)
#   2. REML estimation of SNP heritability in a one-component mixed model
#   3. The effect of ancestry-PC adjustment on the heritability estimate
#   4. Comparison with the realized variance from the planted causal effects
#   5. A profile-likelihood confidence interval
#
###############################################################################


# ---------------------------- 0. Configuration ------------------------------

GRM_MARKER_SET <- "all_qc"  # Alternatives: "all_qc" or "ld_pruned".
PROFILE_GRID <- seq(0.001, 0.999, length.out = 300L)

locate_extension_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  script_arg <- grep("^--file=", args, value = TRUE)
  if (length(script_arg) > 0L) {
    return(dirname(normalizePath(sub("^--file=", "", script_arg[1]))))
  }
  normalizePath(getwd())
}

EXTENSION_DIR <- locate_extension_dir()
source(file.path(EXTENSION_DIR, "gwas_demo_pipeline.R"), chdir = FALSE)

H2_RESULTS_DIR <- file.path(PROJECT_DIR, "results_heritability")
dir.create(H2_RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

h2_section <- function(x) cat("\n", strrep("=", 72), "\n", x, "\n",
                             strrep("=", 72), "\n", sep = "")

# --------------------------- 1. Construct the GRM ---------------------------

h2_section("1. Construct the genomic relationship matrix")

make_grm <- function(genotypes) {
  g <- mean_impute(genotypes)
  allele_frequency <- colMeans(g) / 2
  hwe_sd <- sqrt(2 * allele_frequency * (1 - allele_frequency))
  valid <- is.finite(hwe_sd) & hwe_sd > 0
  if (!all(valid)) {
    g <- g[, valid, drop = FALSE]
    allele_frequency <- allele_frequency[valid]
    hwe_sd <- hwe_sd[valid]
  }

  z <- sweep(g, 2, 2 * allele_frequency, FUN = "-")
  z <- sweep(z, 2, hwe_sd, FUN = "/")
  k <- tcrossprod(z) / ncol(z)
  k <- (k + t(k)) / 2

  # With this normalization, one unit of sigma_g^2 represents approximately
  # one unit of additive genetic variance for an average participant.
  k <- k / mean(diag(k))
  rownames(k) <- colnames(k) <- rownames(genotypes)

  list(K = k, Z = z, n_markers = ncol(z))
}

if (GRM_MARKER_SET == "all_qc") {
  grm_input <- G
} else if (GRM_MARKER_SET == "ld_pruned") {
  grm_input <- G[, pruned_index, drop = FALSE]
} else {
  stop("GRM_MARKER_SET must be 'all_qc' or 'ld_pruned'.")
}

grm <- make_grm(grm_input)
K <- grm$K
cat(sprintf("GRM constructed from %d %s markers.\n",
            grm$n_markers, GRM_MARKER_SET))
cat(sprintf("Mean diagonal = %.3f; median off-diagonal = %.3f.\n",
            mean(diag(K)), median(K[upper.tri(K)])))

# ------------------------- 2. REML helper function --------------------------

fit_grm_reml <- function(y, design, kinship, profile_grid = PROFILE_GRID) {
  design <- as.matrix(design)
  if (qr(design)$rank != ncol(design)) {
    stop("The fixed-effect design matrix is not full rank.")
  }
  if (nrow(design) != length(y) || nrow(kinship) != length(y)) {
    stop("Outcome, design matrix, and GRM dimensions do not agree.")
  }

  kinship <- (kinship + t(kinship)) / 2
  eig <- eigen(kinship, symmetric = TRUE)
  eigenvalues <- pmax(eig$values, 0)
  eigenvectors <- eig$vectors

  y_eigen <- as.numeric(crossprod(eigenvectors, y))
  x_eigen <- crossprod(eigenvectors, design)
  n <- length(y)
  p <- ncol(design)

  # delta = sigma_e^2 / sigma_g^2. Constants that do not depend on delta are
  # omitted because they do not affect optimization or likelihood profiling.
  restricted_nll <- function(log_delta) {
    delta <- exp(log_delta)
    v <- eigenvalues + delta
    inverse_v <- 1 / v
    xt_vinv_x <- crossprod(
      x_eigen,
      sweep(x_eigen, 1, inverse_v, FUN = "*")
    )
    logdet_x <- determinant(xt_vinv_x, logarithm = TRUE)
    if (logdet_x$sign <= 0) return(Inf)

    beta <- tryCatch(
      solve(xt_vinv_x, crossprod(x_eigen, inverse_v * y_eigen)),
      error = function(e) NULL
    )
    if (is.null(beta)) return(Inf)

    residual <- y_eigen - as.numeric(x_eigen %*% beta)
    quadratic <- sum(inverse_v * residual^2)
    if (!is.finite(quadratic) || quadratic <= 0) return(Inf)
    sigma_g2 <- quadratic / (n - p)

    0.5 * ((n - p) * log(sigma_g2) + sum(log(v)) +
             as.numeric(logdet_x$modulus))
  }

  optimum <- optimize(restricted_nll, interval = c(-12, 12), tol = 1e-8)
  delta <- exp(optimum$minimum)
  h2 <- 1 / (1 + delta)

  v <- eigenvalues + delta
  inverse_v <- 1 / v
  xt_vinv_x <- crossprod(
    x_eigen,
    sweep(x_eigen, 1, inverse_v, FUN = "*")
  )
  beta <- solve(xt_vinv_x, crossprod(x_eigen, inverse_v * y_eigen))
  residual <- y_eigen - as.numeric(x_eigen %*% beta)
  sigma_g2 <- sum(inverse_v * residual^2) / (n - p)
  sigma_e2 <- delta * sigma_g2

  profile_nll <- vapply(profile_grid, function(h) {
    restricted_nll(log((1 - h) / h))
  }, numeric(1))
  likelihood_ratio <- 2 * (profile_nll - optimum$objective)
  inside <- is.finite(likelihood_ratio) &
    likelihood_ratio <= qchisq(0.95, df = 1)
  profile_ci <- if (any(inside)) range(profile_grid[inside]) else c(NA, NA)

  list(
    h2 = h2,
    sigma_g2 = sigma_g2,
    sigma_e2 = sigma_e2,
    delta = delta,
    beta = beta,
    restricted_nll = optimum$objective,
    profile = data.frame(
      h2 = profile_grid,
      relative_deviance = likelihood_ratio
    ),
    profile_ci = profile_ci
  )
}

# ---------------------------- 3. Estimate h2 -------------------------------

h2_section("2. Estimate SNP heritability by REML")

y <- pheno$quantitative_trait
X_no_pc <- cbind(Intercept = 1, base_covariates)
X_with_pc <- cbind(Intercept = 1, pc_adjusted_covariates)

cat("Fitting a model without ancestry PCs...\n")
fit_h2_no_pc <- fit_grm_reml(y, X_no_pc, K)
cat("Fitting a model with ancestry PC1...\n")
fit_h2_pc <- fit_grm_reml(y, X_with_pc, K)

heritability_results <- data.frame(
  model = c("Age + sex + batch", "Age + sex + batch + PC1"),
  h2_SNP = c(fit_h2_no_pc$h2, fit_h2_pc$h2),
  profile_CI_low = c(fit_h2_no_pc$profile_ci[1],
                     fit_h2_pc$profile_ci[1]),
  profile_CI_high = c(fit_h2_no_pc$profile_ci[2],
                      fit_h2_pc$profile_ci[2]),
  sigma_g2 = c(fit_h2_no_pc$sigma_g2, fit_h2_pc$sigma_g2),
  sigma_e2 = c(fit_h2_no_pc$sigma_e2, fit_h2_pc$sigma_e2),
  delta = c(fit_h2_no_pc$delta, fit_h2_pc$delta)
)
write.csv(heritability_results,
          file.path(H2_RESULTS_DIR, "snp_heritability_estimates.csv"),
          row.names = FALSE)
print(heritability_results, digits = 4, row.names = FALSE)

# ----------------------- 4. Compare with simulation truth ------------------

h2_section("3. Compare REML with the planted causal effects")

truth_path <- file.path(KEY_DIR, "causal_variants.csv")
oracle_results <- NULL

residualize <- function(v, design) qr.resid(qr(design), v)

if (file.exists(truth_path)) {
  truth <- read.csv(truth_path, check.names = FALSE)
  causal_index <- match(truth$SNP, colnames(G))
  if (anyNA(causal_index)) {
    stop("At least one planted causal SNP was removed or cannot be matched.")
  }

  causal_genotypes <- mean_impute(G[, causal_index, drop = FALSE])
  true_genetic_value <- as.numeric(
    causal_genotypes %*% truth$beta_quantitative
  )

  y_pc_residual <- residualize(y, X_with_pc)
  g_pc_residual <- residualize(true_genetic_value, X_with_pc)

  raw_realized_fraction <- var(true_genetic_value) / var(y)
  conditional_realized_fraction <- var(g_pc_residual) / var(y_pc_residual)

  reduced_sse <- sum(y_pc_residual^2)
  full_design <- cbind(X_with_pc, causal_genotypes)
  full_sse <- sum(qr.resid(qr(full_design), y)^2)
  incremental_r2 <- 1 - full_sse / reduced_sse

  oracle_results <- data.frame(
    quantity = c(
      "Raw realized Var(planted genetic value) / Var(phenotype)",
      "PC-conditional realized variance ratio",
      "Incremental R2 from re-fitting the three causal SNPs"
    ),
    estimate = c(raw_realized_fraction,
                 conditional_realized_fraction,
                 incremental_r2)
  )
  write.csv(oracle_results,
            file.path(H2_RESULTS_DIR, "oracle_causal_variance.csv"),
            row.names = FALSE)
  print(oracle_results, digits = 4, row.names = FALSE)

  png(file.path(H2_RESULTS_DIR, "02_phenotype_vs_true_genetic_value.png"),
      width = 1500, height = 1200, res = 180)
  plot(true_genetic_value, y, pch = 19, cex = 0.75,
       col = adjustcolor("#2166AC", alpha.f = 0.55),
       xlab = "Planted additive genetic value",
       ylab = "Simulated quantitative trait",
       main = "Known causal component versus observed phenotype")
  abline(lm(y ~ true_genetic_value), col = "#B2182B", lwd = 2)
  dev.off()
} else {
  cat("Instructor causal-variant key is absent; skipping the oracle comparison.\n")
}

# ------------------------------ 5. Plots -----------------------------------

h2_section("4. Plot the REML profile likelihood")

profile_deviance <- c(
  fit_h2_no_pc$profile$relative_deviance,
  fit_h2_pc$profile$relative_deviance
)
profile_deviance <- profile_deviance[is.finite(profile_deviance)]
profile_ymax <- min(12, max(profile_deviance))

png(file.path(H2_RESULTS_DIR, "01_heritability_profile.png"),
    width = 1600, height = 1200, res = 180)
plot(fit_h2_no_pc$profile$h2,
     fit_h2_no_pc$profile$relative_deviance,
     type = "l", lwd = 2, col = "#D73027",
     xlab = expression("SNP heritability " * h[SNP]^2),
     ylab = expression(2 * Delta * "REML negative log-likelihood"),
     ylim = c(0, profile_ymax),
     main = "Profile likelihood for SNP heritability")
lines(fit_h2_pc$profile$h2,
      fit_h2_pc$profile$relative_deviance,
      lwd = 2, col = "#2166AC")
abline(h = qchisq(0.95, df = 1), lty = 2, col = "gray35")
abline(v = fit_h2_no_pc$h2, lty = 3, col = "#D73027")
abline(v = fit_h2_pc$h2, lty = 3, col = "#2166AC")
legend("topright",
       legend = c(
         sprintf("Without PC1: h2 = %.2f", fit_h2_no_pc$h2),
         sprintf("With PC1: h2 = %.2f", fit_h2_pc$h2),
         "Approximate 95% cutoff"
       ),
       col = c("#D73027", "#2166AC", "gray35"),
       lty = c(1, 1, 2), lwd = c(2, 2, 1), bty = "n")
dev.off()

###############################################################################
# Caveats:
# - A single GRM component is a teaching simplification. Real studies may use
#   ancestry-stratified analyses, multiple variance components, family designs,
#   or LD-aware methods.
# - Estimates depend on allele-frequency scaling, marker ascertainment, LD,
#   covariates, phenotype definition, and the sampled population.
# - Liability-scale conversion is required for case-control heritability; this
#   script deliberately estimates heritability only for the quantitative trait.
###############################################################################
