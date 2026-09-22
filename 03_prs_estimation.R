###############################################################################
# POLYGENIC RISK/SCORE ESTIMATION 
#

# Demonstrates:
#   1. Why PRS weights must be estimated outside the test set
#   2. Training-only imputation, PCA, and SNP-effect estimation
#   3. Joint SNP selection and shrinkage using LASSO
#   4. LASSO penalty (lambda) selection in a validation set
#   5. Final incremental prediction in an untouched test set
#   6. Comparison with an oracle score using the planted causal effects
#
###############################################################################

# ---------------------------- 0. Configuration ------------------------------

DISCOVERY_FRACTION <- 0.60
VALIDATION_FRACTION <- 0.20
SPLIT_SEED <- 20260920L
PRS_N_PCS <- 1L
PRS_LASSO_ALPHA <- 1
PRS_LASSO_N_LAMBDA <- 100L
PRS_LASSO_ZERO_TOL <- 1e-10

locate_extension_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  script_arg <- grep("^--file=", args, value = TRUE)
  if (length(script_arg) > 0L) {
    return(dirname(normalizePath(sub("^--file=", "", script_arg[1]))))
  }
  normalizePath(getwd())
}

#EXTENSION_DIR <- locate_extension_dir()
#source(file.path(EXTENSION_DIR, "gwas_demo_pipeline.R"), chdir = FALSE)

PRS_RESULTS_DIR <- file.path(PROJECT_DIR, "results_prs")
dir.create(PRS_RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

prs_section <- function(x) cat("\n", strrep("=", 72), "\n", x, "\n",
                               strrep("=", 72), "\n", sep = "")

# ----------------------- 1. Deterministic data split ------------------------

prs_section("1. Create discovery, validation, and test sets")

# A simple reproducible hash avoids relying on R-version-specific random
# sampling. It is used only to order IDs within each ancestry stratum.
stable_id_hash <- function(id, seed = SPLIT_SEED) {
  vapply(id, function(one_id) {
    h <- as.numeric(seed)
    for (code in utf8ToInt(one_id)) {
      h <- (h * 131 + code) %% 2147483647
    }
    h
  }, numeric(1))
}

stratified_split <- function(ids, strata, discovery_fraction,
                             validation_fraction) {
  if (discovery_fraction <= 0 || validation_fraction <= 0 ||
      discovery_fraction + validation_fraction >= 1) {
    stop("Split fractions must be positive and sum to less than one.")
  }
  
  split <- rep(NA_character_, length(ids))
  for (level in unique(strata)) {
    index <- which(strata == level)
    index <- index[order(stable_id_hash(ids[index]))]
    n_discovery <- floor(discovery_fraction * length(index))
    n_validation <- floor(validation_fraction * length(index))
    
    split[index[seq_len(n_discovery)]] <- "discovery"
    validation_positions <- n_discovery + seq_len(n_validation)
    split[index[validation_positions]] <- "validation"
    split[index[(n_discovery + n_validation + 1L):length(index)]] <- "test"
  }
  factor(split, levels = c("discovery", "validation", "test"))
}

analysis_split <- stratified_split(
  rownames(G),
  pheno$self_report_group,
  DISCOVERY_FRACTION,
  VALIDATION_FRACTION
)

discovery_index <- which(analysis_split == "discovery")
validation_index <- which(analysis_split == "validation")
test_index <- which(analysis_split == "test")

split_table <- as.data.frame.matrix(
  table(split = analysis_split, group = pheno$self_report_group)
)
split_table$split <- rownames(split_table)
split_table <- split_table[, c("split", setdiff(names(split_table), "split"))]
write.csv(split_table, file.path(PRS_RESULTS_DIR, "data_split_counts.csv"),
          row.names = FALSE)
print(split_table, row.names = FALSE)

# ---------------- 2. Training-only imputation and ancestry PCA --------------

prs_section("2. Estimate imputation values and PCs in the discovery set")

impute_from_reference <- function(genotypes, reference_rows) {
  reference_means <- colMeans(genotypes[reference_rows, , drop = FALSE],
                              na.rm = TRUE)
  if (any(!is.finite(reference_means))) {
    stop("A SNP is entirely missing in the discovery set.")
  }
  out <- genotypes
  missing <- which(is.na(out), arr.ind = TRUE)
  if (nrow(missing) > 0L) {
    out[missing] <- reference_means[missing[, 2]]
  }
  list(genotypes = out, means = reference_means)
}

imputed <- impute_from_reference(G, discovery_index)
G_prs <- imputed$genotypes

# Select and prune PCA markers using discovery genotypes only.
train_af <- colMeans(G_prs[discovery_index, , drop = FALSE]) / 2
train_maf <- pmin(train_af, 1 - train_af)
prs_pruned_index <- ld_prune(
  G[discovery_index, , drop = FALSE],
  map,
  train_maf,
  window_bp = LD_WINDOW_BP,
  r2_max = LD_R2_MAX
)

G_pca_train <- G_prs[discovery_index, prs_pruned_index, drop = FALSE]
pca_frequency <- colMeans(G_pca_train) / 2
pca_sd <- sqrt(2 * pca_frequency * (1 - pca_frequency))
pca_valid <- is.finite(pca_sd) & pca_sd > 0
G_pca_train <- G_pca_train[, pca_valid, drop = FALSE]
pca_frequency <- pca_frequency[pca_valid]
pca_sd <- pca_sd[pca_valid]

Z_pca_train <- sweep(G_pca_train, 2, 2 * pca_frequency, FUN = "-")
Z_pca_train <- sweep(Z_pca_train, 2, pca_sd, FUN = "/")
pca_train <- prcomp(
  Z_pca_train,
  center = FALSE,
  scale. = FALSE,
  rank. = max(10L, PRS_N_PCS)
)

# Project validation and test participants using discovery-set allele
# frequencies and loadings. No phenotype from those sets is used.
G_pca_all <- G_prs[, prs_pruned_index, drop = FALSE]
G_pca_all <- G_pca_all[, pca_valid, drop = FALSE]
Z_pca_all <- sweep(G_pca_all, 2, 2 * pca_frequency, FUN = "-")
Z_pca_all <- sweep(Z_pca_all, 2, pca_sd, FUN = "/")
pca_rotation <- pca_train$rotation[, seq_len(PRS_N_PCS), drop = FALSE]
prs_pcs <- Z_pca_all %*% pca_rotation
colnames(prs_pcs) <- paste0("PC", seq_len(ncol(prs_pcs)))

cat(sprintf("Training-only PCA used %d LD-pruned common SNPs.\n",
            ncol(Z_pca_train)))

# Covariate scaling is also estimated in the discovery set.
age_mean <- mean(pheno$age[discovery_index])
age_sd <- sd(pheno$age[discovery_index])
prs_pheno <- pheno
prs_pheno$age_z <- (prs_pheno$age - age_mean) / age_sd
prs_pheno$sex <- factor(prs_pheno$sex, levels = sort(unique(pheno$sex)))
prs_pheno$batch <- factor(prs_pheno$batch,
                          levels = sort(unique(pheno$batch)))

prs_base_covariates <- model.matrix(
  ~ age_z + sex + batch,
  data = prs_pheno
)[, -1, drop = FALSE]
prs_covariates <- cbind(
  prs_base_covariates,
  prs_pcs[, seq_len(PRS_N_PCS), drop = FALSE]
)

# ---------------------- 3. Discovery LASSO path ----------------------------

prs_section("3. Fit a LASSO PRS path in the discovery set")

if (!requireNamespace("glmnet", quietly = TRUE)) {
  stop("The LASSO PRS module requires the R package 'glmnet'.")
}

prs_outcome <- pheno$quantitative_trait

# Fit covariates and SNPs jointly, but penalize only the SNP coefficients.
# Thus age/sex/batch/PC adjustment is retained while the L1 penalty performs
# SNP selection and shrinkage.
n_prs_covariates <- ncol(prs_covariates)
lasso_x_all <- cbind(
  prs_covariates,
  G_prs
)

# Prefix names only to make the glmnet coefficient matrix unambiguous.
colnames(lasso_x_all) <- c(
  paste0("COV_", make.names(colnames(prs_covariates), unique = TRUE)),
  paste0("SNP_", colnames(G_prs))
)

lasso_penalty_factor <- c(
  rep(0, n_prs_covariates),  # covariates are not penalized
  rep(1, ncol(G_prs))        # SNP effects receive the L1 penalty
)

lasso_fit <- glmnet::glmnet(
  x = lasso_x_all[discovery_index, , drop = FALSE],
  y = prs_outcome[discovery_index],
  family = "gaussian",
  alpha = PRS_LASSO_ALPHA,
  nlambda = PRS_LASSO_N_LAMBDA,
  standardize = TRUE,
  intercept = TRUE,
  penalty.factor = lasso_penalty_factor
)

lasso_lambdas <- lasso_fit$lambda
lasso_beta_all <- as.matrix(lasso_fit$beta)

# Extract only the SNP coefficients. glmnet returns coefficients on the
# original predictor scale, so these can be multiplied directly by G_prs.
snp_rows <- n_prs_covariates + seq_len(ncol(G_prs))
lasso_snp_beta <- lasso_beta_all[snp_rows, , drop = FALSE]
rownames(lasso_snp_beta) <- colnames(G_prs)

cat(sprintf(
  "Fitted a LASSO path with %d lambda values across %d SNPs.\n",
  length(lasso_lambdas), ncol(G_prs)
))

standardize_from_discovery <- function(score, discovery_rows) {
  center <- mean(score[discovery_rows])
  scale <- sd(score[discovery_rows])
  if (!is.finite(scale) || scale <= 0) {
    return(list(score = rep(0, length(score)), center = center,
                scale = NA_real_, valid = FALSE))
  }
  list(score = (score - center) / scale, center = center,
       scale = scale, valid = TRUE)
}

evaluate_quantitative_prs <- function(rows, score, y, covariates) {
  y_subset <- y[rows]
  x0 <- cbind(Intercept = 1, covariates[rows, , drop = FALSE])
  x1 <- cbind(x0, PRS = score[rows])
  
  fit0 <- lm.fit(x0, y_subset)
  fit1 <- lm.fit(x1, y_subset)
  total_ss <- sum((y_subset - mean(y_subset))^2)
  r2_base <- 1 - sum(fit0$residuals^2) / total_ss
  r2_full <- 1 - sum(fit1$residuals^2) / total_ss
  
  beta <- se <- p_value <- NA_real_
  if (fit1$rank == ncol(x1)) {
    df <- length(y_subset) - fit1$rank
    residual_variance <- sum(fit1$residuals^2) / df
    covariance <- residual_variance * solve(crossprod(x1))
    beta <- unname(tail(fit1$coefficients, 1))
    se <- sqrt(tail(diag(covariance), 1))
    p_value <- 2 * pt(abs(beta / se), df = df, lower.tail = FALSE)
  }
  
  data.frame(
    R2_base = r2_base,
    R2_with_PRS = r2_full,
    incremental_R2 = r2_full - r2_base,
    PRS_beta_per_SD = beta,
    PRS_SE = se,
    PRS_P = p_value,
    phenotype_PRS_correlation = suppressWarnings(cor(y_subset, score[rows]))
  )
}

# ------------------------ 4. Tune LASSO lambda -----------------------------

prs_section("4. Select the LASSO penalty in the validation set")

lambda_rows <- vector("list", length(lasso_lambdas))
selected_by_lambda <- vector("list", length(lasso_lambdas))
score_by_lambda <- vector("list", length(lasso_lambdas))

for (q in seq_along(lasso_lambdas)) {
  lambda <- lasso_lambdas[q]
  beta_q <- lasso_snp_beta[, q]
  selected <- which(abs(beta_q) > PRS_LASSO_ZERO_TOL)
  selected_by_lambda[[q]] <- selected
  
  raw_score <- if (length(selected) > 0L) {
    as.numeric(
      G_prs[, selected, drop = FALSE] %*%
        beta_q[selected]
    )
  } else {
    rep(0, nrow(G_prs))
  }
  
  standardized <- standardize_from_discovery(raw_score, discovery_index)
  score_by_lambda[[q]] <- standardized$score
  
  validation_metrics <- evaluate_quantitative_prs(
    validation_index,
    standardized$score,
    prs_outcome,
    prs_covariates
  )
  
  lambda_rows[[q]] <- cbind(
    data.frame(
      lambda = lambda,
      log_lambda = log(lambda),
      n_selected_SNPs = length(selected),
      valid_score = standardized$valid
    ),
    validation_metrics
  )
}

lambda_results <- do.call(rbind, lambda_rows)
selection_value <- lambda_results$incremental_R2
selection_value[!lambda_results$valid_score | !is.finite(selection_value)] <- -Inf

if (all(selection_value == -Inf)) {
  stop("No LASSO lambda produced a usable PRS.")
}

best_index <- which.max(selection_value)
best_lambda <- lambda_results$lambda[best_index]
best_selected <- selected_by_lambda[[best_index]]
best_score <- score_by_lambda[[best_index]]
best_beta <- lasso_snp_beta[, best_index]

lambda_results$selected_in_validation <- FALSE
lambda_results$selected_in_validation[best_index] <- TRUE

write.csv(
  lambda_results,
  file.path(PRS_RESULTS_DIR, "validation_lasso_results.csv"),
  row.names = FALSE
)
print(lambda_results, digits = 4, row.names = FALSE)

cat(sprintf(
  "Selected lambda %.5g with %d nonzero SNP coefficients.\n",
  best_lambda, length(best_selected)
))

# -------------------------- 5. Final test evaluation ------------------------

prs_section("5. Evaluate the selected PRS in the untouched test set")

test_metrics <- evaluate_quantitative_prs(
  test_index,
  best_score,
  prs_outcome,
  prs_covariates
)
test_metrics <- cbind(
  data.frame(
    score = "LASSO PRS selected in validation",
    lambda = best_lambda,
    n_selected_SNPs = length(best_selected)
  ),
  test_metrics
)

truth_path <- file.path(KEY_DIR, "causal_variants.csv")
oracle_score <- rep(NA_real_, nrow(G_prs))
if (file.exists(truth_path)) {
  truth <- read.csv(truth_path, check.names = FALSE)
  causal_index <- match(truth$SNP, colnames(G_prs))
  if (!anyNA(causal_index)) {
    oracle_raw <- as.numeric(
      G_prs[, causal_index, drop = FALSE] %*% truth$beta_quantitative
    )
    oracle_standardized <- standardize_from_discovery(
      oracle_raw,
      discovery_index
    )
    oracle_score <- oracle_standardized$score
    oracle_metrics <- evaluate_quantitative_prs(
      test_index,
      oracle_score,
      prs_outcome,
      prs_covariates
    )
    oracle_metrics <- cbind(
      data.frame(
        score = "Oracle score using three planted effects",
        lambda = NA_real_,
        n_selected_SNPs = length(causal_index)
      ),
      oracle_metrics
    )
    test_metrics <- rbind(test_metrics, oracle_metrics)
  }
}

write.csv(test_metrics,
          file.path(PRS_RESULTS_DIR, "test_prediction_results.csv"),
          row.names = FALSE)
print(test_metrics, digits = 4, row.names = FALSE)

selected_weights <- data.frame(
  map[best_selected, , drop = FALSE],
  lasso_beta = best_beta[best_selected],
  abs_lasso_beta = abs(best_beta[best_selected]),
  score_weight = best_beta[best_selected]
)
if (file.exists(truth_path)) {
  selected_weights$KNOWN_CAUSAL <- selected_weights$SNP %in% truth$SNP
}
selected_weights <- selected_weights[
  order(selected_weights$abs_lasso_beta, decreasing = TRUE),
]
write.csv(
  selected_weights,
  file.path(PRS_RESULTS_DIR, "selected_lasso_prs_weights.csv"),
  row.names = FALSE
)

individual_scores <- data.frame(
  IID = rownames(G_prs),
  split = analysis_split,
  self_report_group = pheno$self_report_group,
  quantitative_trait = prs_outcome,
  PC1_training_projection = prs_pcs[, 1],
  PRS_standardized = best_score,
  oracle_score_standardized = oracle_score
)
write.csv(individual_scores,
          file.path(PRS_RESULTS_DIR, "individual_prs_scores.csv"),
          row.names = FALSE)

# -------------------------------- 6. Plots ---------------------------------

prs_section("6. Create PRS teaching plots")

plot_order <- order(log10(lambda_results$lambda))
png(file.path(PRS_RESULTS_DIR, "01_validation_lasso_tuning.png"),
    width = 1600, height = 1200, res = 180)
plot(log10(lambda_results$lambda[plot_order]),
     lambda_results$incremental_R2[plot_order],
     type = "b", pch = 19, lwd = 2, col = "#2166AC",
     xlab = expression(log[10]("LASSO lambda")),
     ylab = expression("Validation incremental " * R^2),
     main = "Choose the LASSO penalty without using the test set")
points(log10(best_lambda),
       lambda_results$incremental_R2[best_index],
       pch = 23, cex = 1.6, bg = "#FFD92F", col = "black")
text(log10(lambda_results$lambda[plot_order]),
     lambda_results$incremental_R2[plot_order],
     labels = lambda_results$n_selected_SNPs[plot_order],
     pos = 3, cex = 0.72)
legend("bottomright", legend = "Labels = number of nonzero SNP coefficients",
       bty = "n")
dev.off()

test_design <- cbind(Intercept = 1,
                     prs_covariates[test_index, , drop = FALSE])
test_y_residual <- qr.resid(qr(test_design), prs_outcome[test_index])
test_prs_residual <- qr.resid(qr(test_design), best_score[test_index])

png(file.path(PRS_RESULTS_DIR, "02_test_PRS_prediction.png"),
    width = 1500, height = 1200, res = 180)
plot(test_prs_residual, test_y_residual, pch = 19, cex = 0.85,
     col = adjustcolor("#2166AC", alpha.f = 0.60),
     xlab = "Covariate-adjusted PRS in the test set",
     ylab = "Covariate-adjusted quantitative trait",
     main = "Held-out PRS evaluation")
abline(lm(test_y_residual ~ test_prs_residual),
       col = "#B2182B", lwd = 2)
dev.off()


###############################################################################
# Production caveats:
# - Real PRS development requires external GWAS weights or much larger discovery
#   samples, ancestry-matched LD information, harmonized alleles/builds, and
#   independent external validation.
# - LASSO estimates SNP effects jointly and performs variable selection, but
#   correlated predictors can make the selected SNP set unstable. Elastic net,
#   LD-aware shrinkage, and Bayesian PRS methods are useful extensions.
# - This simulation has only three planted causal loci, so it is intentionally
#   much sparser and simpler than a genuinely polygenic human trait.
###############################################################################
