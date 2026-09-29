###############################################################################
# GENE-ENVIRONMENT INTERACTION 
#
# This script uses the same cohort, genotype coding, sample QC, variant QC,
# LD pruning, and ancestry PCs as gwas_demo_pipeline.R. A separate phenotype
# file adds a binary environmental exposure and two outcomes with a planted
# interaction at rs0001084.
#
# Demonstrates:
#   1. Alignment and QC of an environmental exposure
#   2. The model Y = beta0 + betaG*G + betaE*E + betaGE*G*E + covariates
#   3. A one-degree-of-freedom interaction test for betaGE = 0
#   4. A two-degree-of-freedom joint test of betaG = betaGE = 0
#   5. Genotype-by-exposure cell-count checks and G-E correlation checks
#   6. PC adjustment, robust HC3 standard errors, and multiple testing
#   7. QQ, Manhattan, and exposure-stratified effect plots
#
# Set GXE_TRAIT_TYPE to "binary" for the optional logistic-regression version.
###############################################################################

options(stringsAsFactors = FALSE)

# ---------------------------- 0. Configuration ------------------------------

GXE_TRAIT_TYPE <- "quantitative"  # "quantitative" or "binary"
GXE_TARGET_SNP <- "rs0001084"
GXE_N_PCS <- 1L
GXE_MIN_MINOR_ALLELE_COPIES_PER_EXPOSURE_GROUP <- 10L
GXE_ALPHA <- 0.05


GXE_RESULTS_DIR <- file.path(PROJECT_DIR, "results_gxe")
dir.create(GXE_RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

gxe_section <- function(x) cat("\n", strrep("=", 72), "\n", x, "\n",
                               strrep("=", 72), "\n", sep = "")

# ---------------------- 1. Import the GxE phenotypes -------------------------

gxe_section("1. Import and align exposure and outcome data")

gxe_path <- file.path(DATA_DIR, "gxe_phenotypes.csv")
if (!file.exists(gxe_path)) {
  stop("Missing data/gxe_phenotypes.csv. Re-extract the complete teaching bundle.")
}

gxe_pheno <- read.csv(gxe_path, check.names = FALSE)
required_gxe_columns <- c(
  "IID", "high_exposure", "gxe_quantitative_trait", "gxe_case_status"
)
if (!all(required_gxe_columns %in% names(gxe_pheno))) {
  stop("The GxE phenotype file is missing required columns: ",
       paste(setdiff(required_gxe_columns, names(gxe_pheno)), collapse = ", "))
}
if (anyDuplicated(gxe_pheno$IID)) stop("The GxE phenotype file has duplicated IIDs.")
if (!all(pheno$IID %in% gxe_pheno$IID)) {
  stop("Some post-QC genotype samples are absent from the GxE phenotype file.")
}

# Match the new file to the post-QC sample order produced by the core pipeline.
gxe_pheno <- gxe_pheno[match(pheno$IID, gxe_pheno$IID), , drop = FALSE]
stopifnot(identical(pheno$IID, gxe_pheno$IID))

environment <- as.numeric(gxe_pheno$high_exposure)
if (any(!environment %in% c(0, 1), na.rm = TRUE)) {
  stop("high_exposure must be coded 0, 1, or NA.")
}

if (GXE_TRAIT_TYPE == "quantitative") {
  gxe_outcome <- gxe_pheno$gxe_quantitative_trait
  outcome_label <- "GxE quantitative trait"
} else if (GXE_TRAIT_TYPE == "binary") {
  gxe_outcome <- gxe_pheno$gxe_case_status
  outcome_label <- "GxE case status"
  if (any(!gxe_outcome %in% c(0, 1), na.rm = TRUE)) {
    stop("gxe_case_status must be coded 0, 1, or NA.")
  }
} else {
  stop("GXE_TRAIT_TYPE must be 'quantitative' or 'binary'.")
}

analysis_rows <- which(complete.cases(
  gxe_outcome,
  environment,
  pheno[, c("age", "sex", "batch")]
))
if (length(analysis_rows) < nrow(G)) {
  cat(sprintf("Complete-case analysis removed %d participants.\n",
              nrow(G) - length(analysis_rows)))
}

G_gxe <- G[analysis_rows, , drop = FALSE]
pheno_gxe <- pheno[analysis_rows, , drop = FALSE]
gxe_pheno <- gxe_pheno[analysis_rows, , drop = FALSE]
environment <- environment[analysis_rows]
gxe_outcome <- gxe_outcome[analysis_rows]
if (GXE_N_PCS < 1L || GXE_N_PCS > ncol(pcs_for_model)) {
  stop("GXE_N_PCS must be between 1 and ", ncol(pcs_for_model),
       " for this teaching dataset.")
}
gxe_pcs <- pcs_for_model[analysis_rows, seq_len(GXE_N_PCS), drop = FALSE]

cat(sprintf("GxE analysis includes %d participants: %d unexposed and %d exposed.\n",
            length(environment), sum(environment == 0), sum(environment == 1)))
if (GXE_TRAIT_TYPE == "binary") {
  cat(sprintf("Cases = %d; controls = %d.\n",
              sum(gxe_outcome == 1), sum(gxe_outcome == 0)))
}

exposure_summary <- as.data.frame.matrix(table(
  self_report_group = pheno_gxe$self_report_group,
  high_exposure = environment
))
exposure_summary$self_report_group <- rownames(exposure_summary)
rownames(exposure_summary) <- NULL
names(exposure_summary)[names(exposure_summary) == "0"] <- "unexposed"
names(exposure_summary)[names(exposure_summary) == "1"] <- "exposed"
exposure_summary$exposure_prevalence <- with(
  exposure_summary,
  exposed / (unexposed + exposed)
)
write.csv(exposure_summary,
          file.path(GXE_RESULTS_DIR, "exposure_summary.csv"), row.names = FALSE)
print(exposure_summary, digits = 3, row.names = FALSE)

# ------------------ 2. Build covariates and interaction QC -----------------

gxe_section("2. Build covariates and inspect genotype-by-exposure support")

# Main effects for both G and E must accompany G:E. Age is centered and scaled
# so the intercept corresponds to a participant of average age.
age_mean <- mean(pheno_gxe$age)
age_sd <- sd(pheno_gxe$age)
gxe_model_data <- pheno_gxe
gxe_model_data$age_z <- (gxe_model_data$age - age_mean) / age_sd
gxe_model_data$sex <- factor(gxe_model_data$sex)
gxe_model_data$batch <- factor(gxe_model_data$batch)

measured_covariates <- model.matrix(
  ~ age_z + sex + batch,
  data = gxe_model_data
)[, -1, drop = FALSE]
colnames(gxe_pcs) <- paste0("PC", seq_len(ncol(gxe_pcs)))

# E belongs in the baseline design because the interaction model always
# includes the environmental main effect.
covariates_no_pc <- cbind(E = environment, measured_covariates)
covariates_with_pc <- cbind(E = environment, measured_covariates, gxe_pcs)

G_scan <- mean_impute(G_gxe)
allele_frequency <- colMeans(G_scan) / 2
minor_allele_frequency <- pmin(allele_frequency, 1 - allele_frequency)

# Count minor-allele copies separately within exposure groups. Interaction
# estimates become unstable when one group has almost no minor alleles.
G_minor <- G_gxe
flip_to_minor <- allele_frequency > 0.5
if (any(flip_to_minor)) {
  G_minor[, flip_to_minor] <- 2 - G_minor[, flip_to_minor, drop = FALSE]
}
minor_copies_unexposed <- colSums(
  G_minor[environment == 0, , drop = FALSE], na.rm = TRUE
)
minor_copies_exposed <- colSums(
  G_minor[environment == 1, , drop = FALSE], na.rm = TRUE
)
interaction_testable <-
  minor_copies_unexposed >= GXE_MIN_MINOR_ALLELE_COPIES_PER_EXPOSURE_GROUP &
  minor_copies_exposed >= GXE_MIN_MINOR_ALLELE_COPIES_PER_EXPOSURE_GROUP

cat(sprintf(
  "%d of %d post-QC variants have at least %d minor-allele copies in each exposure group.\n",
  sum(interaction_testable), length(interaction_testable),
  GXE_MIN_MINOR_ALLELE_COPIES_PER_EXPOSURE_GROUP
))

target_index <- match(GXE_TARGET_SNP, map$SNP)
if (is.na(target_index)) stop("The target SNP did not pass variant QC.")

target_cell_counts <- as.data.frame(table(
  genotype = factor(G_gxe[, target_index], levels = 0:2),
  exposure = factor(environment, levels = 0:1)
))
write.csv(target_cell_counts,
          file.path(GXE_RESULTS_DIR, "target_genotype_exposure_counts.csv"),
          row.names = FALSE)
cat("Observed target-SNP genotype counts by exposure group:\n")
print(target_cell_counts, row.names = FALSE)

# ------------------------ 3. Genome-wide test functions --------------------

safe_inverse <- function(x) {
  tryCatch(solve(x), error = function(e) NULL)
}

linear_gxe_scan <- function(y, genotypes, environment, covariates,
                            testable = rep(TRUE, ncol(genotypes))) {
  x0 <- cbind(Intercept = 1, as.matrix(covariates))
  base_fit <- lm.fit(x = x0, y = y)
  base_rss <- sum(base_fit$residuals^2)

  column_names <- c(
    "BETA_G", "SE_G", "P_G", "BETA_GXE", "SE_GXE_MODEL",
    "P_GXE_MODEL", "SE_GXE_HC3", "P_GXE_HC3", "P_JOINT_2DF", "N"
  )
  out <- matrix(NA_real_, nrow = ncol(genotypes), ncol = length(column_names),
                dimnames = list(NULL, column_names))

  for (j in which(testable)) {
    g <- genotypes[, j]
    x <- cbind(x0, SNP = g, GXE = g * environment)
    fit <- lm.fit(x = x, y = y)
    if (fit$rank < ncol(x)) next

    xtx_inverse <- safe_inverse(crossprod(x))
    if (is.null(xtx_inverse)) next

    residuals <- fit$residuals
    residual_ss <- sum(residuals^2)
    df <- length(y) - ncol(x)
    model_covariance <- residual_ss / df * xtx_inverse
    model_se <- sqrt(diag(model_covariance))

    beta_g <- unname(fit$coefficients[ncol(x) - 1L])
    beta_gxe <- unname(fit$coefficients[ncol(x)])
    se_g <- model_se[ncol(x) - 1L]
    se_gxe <- model_se[ncol(x)]
    p_g <- 2 * pt(abs(beta_g / se_g), df = df, lower.tail = FALSE)
    p_gxe <- 2 * pt(abs(beta_gxe / se_gxe), df = df, lower.tail = FALSE)

    # HC3 protects the interaction test against exposure-dependent residual
    # variance. This is the primary quantitative-trait interaction p-value.
    leverage <- rowSums((x %*% xtx_inverse) * x)
    hc3_residual <- residuals / pmax(1 - leverage,
                                     sqrt(.Machine$double.eps))
    weighted_x <- x * as.numeric(hc3_residual)
    hc3_covariance <- xtx_inverse %*% crossprod(weighted_x) %*% xtx_inverse
    hc3_se <- sqrt(diag(hc3_covariance))
    se_gxe_hc3 <- hc3_se[ncol(x)]
    p_gxe_hc3 <- 2 * pt(abs(beta_gxe / se_gxe_hc3), df = df,
                        lower.tail = FALSE)

    # Joint test: compare E + covariates against E + covariates + G + G:E.
    joint_f <- ((base_rss - residual_ss) / 2) / (residual_ss / df)
    p_joint <- pf(max(joint_f, 0), df1 = 2, df2 = df, lower.tail = FALSE)

    out[j, ] <- c(beta_g, se_g, p_g, beta_gxe, se_gxe, p_gxe,
                  se_gxe_hc3, p_gxe_hc3, p_joint, length(y))
  }
  as.data.frame(out)
}

logistic_gxe_scan <- function(y, genotypes, environment, covariates,
                              testable = rep(TRUE, ncol(genotypes))) {
  x0 <- cbind(Intercept = 1, as.matrix(covariates))
  base_fit <- suppressWarnings(glm.fit(x = x0, y = y, family = binomial()))
  base_deviance <- base_fit$deviance

  column_names <- c(
    "BETA_G", "SE_G", "P_G", "BETA_GXE", "SE_GXE_MODEL",
    "P_GXE_MODEL", "SE_GXE_HC3", "P_GXE_HC3", "P_JOINT_2DF", "N"
  )
  out <- matrix(NA_real_, nrow = ncol(genotypes), ncol = length(column_names),
                dimnames = list(NULL, column_names))

  for (j in which(testable)) {
    g <- genotypes[, j]
    x <- cbind(x0, SNP = g, GXE = g * environment)
    fit <- suppressWarnings(glm.fit(x = x, y = y, family = binomial()))
    if (!fit$converged || fit$rank < ncol(x) ||
        any(!is.finite(fit$coefficients))) next

    information <- crossprod(x * sqrt(fit$weights))
    covariance <- safe_inverse(information)
    if (is.null(covariance)) next
    standard_error <- sqrt(diag(covariance))

    beta_g <- unname(fit$coefficients[ncol(x) - 1L])
    beta_gxe <- unname(fit$coefficients[ncol(x)])
    se_g <- standard_error[ncol(x) - 1L]
    se_gxe <- standard_error[ncol(x)]
    p_g <- 2 * pnorm(abs(beta_g / se_g), lower.tail = FALSE)
    p_gxe <- 2 * pnorm(abs(beta_gxe / se_gxe), lower.tail = FALSE)
    likelihood_ratio <- max(base_deviance - fit$deviance, 0)
    p_joint <- pchisq(likelihood_ratio, df = 2, lower.tail = FALSE)

    out[j, ] <- c(beta_g, se_g, p_g, beta_gxe, se_gxe, p_gxe,
                  NA_real_, NA_real_, p_joint, length(y))
  }
  as.data.frame(out)
}

run_gxe_scan <- function(y, genotypes, environment, covariates, trait_type,
                         testable) {
  if (trait_type == "quantitative") {
    linear_gxe_scan(y, genotypes, environment, covariates, testable)
  } else {
    logistic_gxe_scan(y, genotypes, environment, covariates, testable)
  }
}

# --------------------- 4. Genome-wide GxE association scan -----------------

gxe_section("3. Run genome-wide interaction tests")

cat("Running the covariate-adjusted GxE scan without ancestry PCs...\n")
gxe_fit_no_pc <- run_gxe_scan(
  gxe_outcome, G_scan, environment, covariates_no_pc,
  GXE_TRAIT_TYPE, interaction_testable
)

cat("Running the covariate- and PC-adjusted GxE scan...\n")
gxe_fit_with_pc <- run_gxe_scan(
  gxe_outcome, G_scan, environment, covariates_with_pc,
  GXE_TRAIT_TYPE, interaction_testable
)

primary_p_no_pc <- if (GXE_TRAIT_TYPE == "quantitative") {
  gxe_fit_no_pc$P_GXE_HC3
} else {
  gxe_fit_no_pc$P_GXE_MODEL
}
primary_p_with_pc <- if (GXE_TRAIT_TYPE == "quantitative") {
  gxe_fit_with_pc$P_GXE_HC3
} else {
  gxe_fit_with_pc$P_GXE_MODEL
}

gxe_results <- cbind(
  map,
  MAF = minor_allele_frequency,
  MINOR_COPIES_UNEXPOSED = minor_copies_unexposed,
  MINOR_COPIES_EXPOSED = minor_copies_exposed,
  INTERACTION_TESTED = interaction_testable,
  BETA_G = gxe_fit_with_pc$BETA_G,
  SE_G = gxe_fit_with_pc$SE_G,
  P_G = gxe_fit_with_pc$P_G,
  BETA_GXE = gxe_fit_with_pc$BETA_GXE,
  SE_GXE_MODEL = gxe_fit_with_pc$SE_GXE_MODEL,
  P_GXE_MODEL = gxe_fit_with_pc$P_GXE_MODEL,
  SE_GXE_HC3 = gxe_fit_with_pc$SE_GXE_HC3,
  P_GXE_HC3 = gxe_fit_with_pc$P_GXE_HC3,
  P_GXE_NO_PC = primary_p_no_pc,
  P_GXE = primary_p_with_pc,
  P_JOINT_2DF = gxe_fit_with_pc$P_JOINT_2DF
)

n_interaction_tests <- sum(is.finite(gxe_results$P_GXE))
bonferroni_threshold <- GXE_ALPHA / n_interaction_tests
gxe_results$BONFERRONI_PASS <-
  gxe_results$P_GXE < bonferroni_threshold
gxe_results$FDR <- p.adjust(gxe_results$P_GXE, method = "BH")
gxe_results$FDR_PASS <- gxe_results$FDR < 0.05
gxe_results$TARGET_SNP <- gxe_results$SNP == GXE_TARGET_SNP

lambda_no_pc <- genomic_lambda(gxe_results$P_GXE_NO_PC)
lambda_with_pc <- genomic_lambda(gxe_results$P_GXE)

cat(sprintf("Interaction tests performed: %d.\n", n_interaction_tests))
cat(sprintf("Bonferroni threshold: %.3g.\n", bonferroni_threshold))
cat(sprintf("Interaction lambda: %.3f without PCs; %.3f with PCs.\n",
            lambda_no_pc, lambda_with_pc))

gxe_results <- gxe_results[order(gxe_results$P_GXE), ]
write.csv(gxe_results,
          file.path(GXE_RESULTS_DIR,
                    paste0("gxe_results_", GXE_TRAIT_TYPE, ".csv")),
          row.names = FALSE)
write.csv(head(gxe_results, 25),
          file.path(GXE_RESULTS_DIR,
                    paste0("top_gxe_hits_", GXE_TRAIT_TYPE, ".csv")),
          row.names = FALSE)

cat("Top PC-adjusted interaction results:\n")
print(head(gxe_results[, c(
  "SNP", "CHR", "BP", "BETA_GXE", "P_GXE", "P_JOINT_2DF",
  "BONFERRONI_PASS", "TARGET_SNP"
)], 10), digits = 4, row.names = FALSE)

# ------------------ 5. Interpret the planted target interaction ------------

gxe_section("4. Interpret the target SNP and check G-E correlation")

target_g <- G_scan[, target_index]
target_x0 <- cbind(Intercept = 1, covariates_with_pc)
target_x <- cbind(target_x0, SNP = target_g, GXE = target_g * environment)

if (GXE_TRAIT_TYPE == "quantitative") {
  target_fit <- lm.fit(x = target_x, y = gxe_outcome)
  target_df <- length(gxe_outcome) - ncol(target_x)
  target_residual_ss <- sum(target_fit$residuals^2)
  target_inverse <- solve(crossprod(target_x))
  target_covariance <- target_residual_ss / target_df * target_inverse
  target_reference <- "mean difference per additional effect allele"
} else {
  target_fit <- suppressWarnings(glm.fit(
    x = target_x, y = gxe_outcome, family = binomial()
  ))
  target_df <- Inf
  target_information <- crossprod(target_x * sqrt(target_fit$weights))
  target_covariance <- solve(target_information)
  target_reference <- "log odds ratio per additional effect allele"
}

target_beta <- target_fit$coefficients
target_se <- sqrt(diag(target_covariance))
target_statistic <- target_beta / target_se
target_p <- if (GXE_TRAIT_TYPE == "quantitative") {
  2 * pt(abs(target_statistic), df = target_df, lower.tail = FALSE)
} else {
  2 * pnorm(abs(target_statistic), lower.tail = FALSE)
}

target_model_results <- data.frame(
  term = c("environment", "genotype", "genotype_x_environment"),
  estimate = target_beta[c(2, ncol(target_x) - 1L, ncol(target_x))],
  standard_error = target_se[c(2, ncol(target_x) - 1L, ncol(target_x))],
  p_value = target_p[c(2, ncol(target_x) - 1L, ncol(target_x))],
  effect_scale = target_reference,
  row.names = NULL
)
if (GXE_TRAIT_TYPE == "binary") {
  target_model_results$odds_ratio <- exp(target_model_results$estimate)
}
write.csv(target_model_results,
          file.path(GXE_RESULTS_DIR, "target_snp_model.csv"), row.names = FALSE)
print(target_model_results, digits = 4, row.names = FALSE)

# With E coded 0/1, betaG is the allelic effect among the unexposed and
# betaG + betaGE is the allelic effect among the exposed.
beta_g_position <- ncol(target_x) - 1L
beta_gxe_position <- ncol(target_x)
stratified_contrasts <- rbind(
  unexposed = c(0, 1, 0),
  exposed = c(0, 1, 1)
)
stratified_rows <- lapply(seq_len(nrow(stratified_contrasts)), function(k) {
  contrast <- rep(0, length(target_beta))
  contrast[beta_g_position] <- stratified_contrasts[k, 2]
  contrast[beta_gxe_position] <- stratified_contrasts[k, 3]
  estimate <- sum(contrast * target_beta)
  se <- sqrt(drop(t(contrast) %*% target_covariance %*% contrast))
  statistic <- estimate / se
  p_value <- if (GXE_TRAIT_TYPE == "quantitative") {
    2 * pt(abs(statistic), df = target_df, lower.tail = FALSE)
  } else {
    2 * pnorm(abs(statistic), lower.tail = FALSE)
  }
  data.frame(
    exposure_group = rownames(stratified_contrasts)[k],
    allelic_effect = estimate,
    standard_error = se,
    p_value = p_value,
    odds_ratio = if (GXE_TRAIT_TYPE == "binary") exp(estimate) else NA_real_
  )
})
stratified_effects <- do.call(rbind, stratified_rows)
write.csv(stratified_effects,
          file.path(GXE_RESULTS_DIR, "target_exposure_stratified_effects.csv"),
          row.names = FALSE)
cat("Target-SNP allelic effects within exposure strata:\n")
print(stratified_effects, digits = 4, row.names = FALSE)

# Gene-environment correlation check: is the exposure associated with the
# target genotype after measured covariates and ancestry PCs are included?
ge_covariates <- cbind(measured_covariates, gxe_pcs)
ge_x <- cbind(Intercept = 1, ge_covariates, SNP = target_g)
ge_fit <- suppressWarnings(glm.fit(
  x = ge_x, y = environment, family = binomial()
))
ge_information <- crossprod(ge_x * sqrt(ge_fit$weights))
ge_covariance <- safe_inverse(ge_information)
if (!ge_fit$converged || is.null(ge_covariance)) {
  ge_result <- data.frame(
    SNP = GXE_TARGET_SNP, log_odds_ratio = NA_real_, standard_error = NA_real_,
    odds_ratio = NA_real_, p_value = NA_real_
  )
} else {
  ge_beta <- tail(ge_fit$coefficients, 1)
  ge_se <- sqrt(tail(diag(ge_covariance), 1))
  ge_result <- data.frame(
    SNP = GXE_TARGET_SNP,
    log_odds_ratio = unname(ge_beta),
    standard_error = unname(ge_se),
    odds_ratio = unname(exp(ge_beta)),
    p_value = unname(2 * pnorm(abs(ge_beta / ge_se), lower.tail = FALSE))
  )
}
write.csv(ge_result,
          file.path(GXE_RESULTS_DIR, "target_gene_environment_correlation.csv"),
          row.names = FALSE)
cat("Adjusted target genotype-exposure association:\n")
print(ge_result, digits = 4, row.names = FALSE)

# ------------------------------- 6. Plots ----------------------------------

gxe_section("5. Create diagnostic and interaction plots")

group_colors <- c(Group_A = "#2166AC", Group_B = "#B2182B")
png(file.path(GXE_RESULTS_DIR, "01_exposure_by_ancestry_group.png"),
    width = 1500, height = 1100, res = 180)
barplot(
  exposure_summary$exposure_prevalence,
  names.arg = exposure_summary$self_report_group,
  col = group_colors[exposure_summary$self_report_group],
  ylim = c(0, 1),
  ylab = "Proportion with high exposure",
  xlab = "Simulated population group",
  main = "Environmental exposure differs across population groups"
)
dev.off()

qq_no_pc <- qq_coordinates(gxe_results$P_GXE_NO_PC)
qq_with_pc <- qq_coordinates(gxe_results$P_GXE)
qq_limit <- max(c(qq_no_pc$expected, qq_no_pc$observed,
                  qq_with_pc$observed), finite = TRUE)
png(file.path(GXE_RESULTS_DIR, "02_gxe_qq_plot.png"),
    width = 1600, height = 1200, res = 180)
plot(qq_no_pc$expected, qq_no_pc$observed, pch = 19, cex = 0.6,
     col = adjustcolor("#D73027", alpha.f = 0.55),
     xlim = c(0, qq_limit), ylim = c(0, qq_limit),
     xlab = expression(Expected~~-log[10](italic(P)[G%*%E])),
     ylab = expression(Observed~~-log[10](italic(P)[G%*%E])),
     main = "QQ plot for genome-wide interaction tests")
points(qq_with_pc$expected, qq_with_pc$observed, pch = 19, cex = 0.6,
       col = adjustcolor("#2166AC", alpha.f = 0.65))
abline(0, 1, lty = 2, col = "gray35")
legend("topleft",
       legend = c(sprintf("No PCs (lambda = %.2f)", lambda_no_pc),
                  sprintf("With PCs (lambda = %.2f)", lambda_with_pc)),
       col = c("#D73027", "#2166AC"), pch = 19, bty = "n")
dev.off()

manhattan <- make_manhattan_data(gxe_results)
md <- manhattan$data
chromosome_colors <- rep(c("#2166AC", "#67A9CF"), 22)
png(file.path(GXE_RESULTS_DIR, "03_gxe_manhattan.png"),
    width = 2200, height = 1100, res = 180)
plot(md$BP_cumulative, -log10(md$P_GXE), pch = 20, cex = 0.65,
     col = chromosome_colors[md$CHR], xaxt = "n",
     xlab = "Chromosome", ylab = expression(-log[10](italic(P)[G%*%E])),
     main = paste("PC-adjusted interaction scan:", outcome_label))
axis(1, at = manhattan$axis, labels = names(manhattan$axis), cex.axis = 0.75)
abline(h = -log10(bonferroni_threshold), col = "#B2182B",
       lty = 2, lwd = 1.5)
target_row <- which(md$SNP == GXE_TARGET_SNP)
points(md$BP_cumulative[target_row], -log10(md$P_GXE[target_row]),
       pch = 23, cex = 1.5, bg = "#FFD92F", col = "black")
text(md$BP_cumulative[target_row], -log10(md$P_GXE[target_row]),
     labels = GXE_TARGET_SNP, pos = 3, cex = 0.7)
dev.off()

# Raw cell means make the interaction visually transparent. The regression
# model above supplies the adjusted inference.
observed_target_g <- G_gxe[, target_index]
cell_rows <- list()
row_number <- 1L
for (e in 0:1) {
  for (dose in 0:2) {
    index <- which(environment == e & observed_target_g == dose)
    if (length(index) == 0L) next
    cell_rows[[row_number]] <- data.frame(
      exposure = e,
      genotype = dose,
      mean = mean(gxe_outcome[index]),
      standard_error = sd(gxe_outcome[index]) / sqrt(length(index)),
      n = length(index)
    )
    row_number <- row_number + 1L
  }
}
cell_summary <- do.call(rbind, cell_rows)
write.csv(cell_summary,
          file.path(GXE_RESULTS_DIR, "target_observed_cell_means.csv"),
          row.names = FALSE)

exposure_colors <- c("0" = "#2166AC", "1" = "#D73027")
y_range <- range(
  cell_summary$mean - 1.96 * cell_summary$standard_error,
  cell_summary$mean + 1.96 * cell_summary$standard_error,
  finite = TRUE
)
png(file.path(GXE_RESULTS_DIR, "04_target_interaction_plot.png"),
    width = 1500, height = 1200, res = 180)
plot(0:2, rep(NA_real_, 3), xlim = c(-0.1, 2.1), ylim = y_range,
     xlab = paste(GXE_TARGET_SNP, "effect-allele dosage"),
     ylab = if (GXE_TRAIT_TYPE == "quantitative")
       "Mean quantitative trait" else "Observed case proportion",
     main = "The genetic effect changes with environmental exposure",
     xaxt = "n")
axis(1, at = 0:2)
for (e in 0:1) {
  rows <- cell_summary[cell_summary$exposure == e, ]
  lines(rows$genotype, rows$mean, type = "b", pch = 19, lwd = 2,
        col = exposure_colors[as.character(e)])
  arrows(rows$genotype,
         rows$mean - 1.96 * rows$standard_error,
         rows$genotype,
         rows$mean + 1.96 * rows$standard_error,
         angle = 90, code = 3, length = 0.05,
         col = exposure_colors[as.character(e)])
}
legend("topleft", legend = c("Unexposed", "Exposed"),
       col = exposure_colors, pch = 19, lwd = 2, bty = "n")
dev.off()

png(file.path(GXE_RESULTS_DIR, "05_main_effect_vs_interaction.png"),
    width = 1500, height = 1250, res = 180)
plot(-log10(gxe_results$P_G), -log10(gxe_results$P_GXE),
     pch = 19, cex = 0.65, col = adjustcolor("gray25", alpha.f = 0.45),
     xlab = expression(-log[10](italic(P)[G])),
     ylab = expression(-log[10](italic(P)[G%*%E])),
     main = "Main-effect and interaction tests answer different questions")
target_in_results <- which(gxe_results$SNP == GXE_TARGET_SNP)
points(-log10(gxe_results$P_G[target_in_results]),
       -log10(gxe_results$P_GXE[target_in_results]),
       pch = 23, cex = 1.5, bg = "#FFD92F", col = "black")
text(-log10(gxe_results$P_G[target_in_results]),
     -log10(gxe_results$P_GXE[target_in_results]),
     labels = GXE_TARGET_SNP, pos = 3, cex = 0.7)
dev.off()

# ------------------------------ 7. Summary ---------------------------------

gxe_section("6. Save and summarize")

target_result <- gxe_results[gxe_results$SNP == GXE_TARGET_SNP, ]
summary_lines <- c(
  "Gene-environment interaction teaching demonstration",
  paste0("Trait type: ", GXE_TRAIT_TYPE),
  sprintf("Analysis sample size: %d", nrow(G_gxe)),
  sprintf("Exposed participants: %d (%.1f%%)", sum(environment == 1),
          100 * mean(environment == 1)),
  sprintf("Variants tested for interaction: %d", n_interaction_tests),
  sprintf("Interaction lambda without ancestry PCs: %.3f", lambda_no_pc),
  sprintf("Interaction lambda with ancestry PCs: %.3f", lambda_with_pc),
  sprintf("Bonferroni threshold: %.6g", bonferroni_threshold),
  sprintf("Bonferroni-significant interaction variants: %d",
          sum(gxe_results$BONFERRONI_PASS, na.rm = TRUE)),
  sprintf("FDR-significant interaction variants: %d",
          sum(gxe_results$FDR_PASS, na.rm = TRUE)),
  "",
  paste0("Target SNP: ", GXE_TARGET_SNP),
  sprintf("Target interaction estimate: %.4f", target_result$BETA_GXE),
  sprintf("Target interaction p-value: %.4g", target_result$P_GXE),
  sprintf("Target 2-df joint p-value: %.4g", target_result$P_JOINT_2DF),
  "",
  "Interpretation when E is coded 0/1:",
  "  betaG is the allelic effect among unexposed participants.",
  "  betaGE is the change in that allelic effect among exposed participants.",
  "  betaG + betaGE is the allelic effect among exposed participants."
)

writeLines(summary_lines,
           file.path(GXE_RESULTS_DIR,
                     paste0("gxe_analysis_summary_", GXE_TRAIT_TYPE, ".txt")))
cat(paste(summary_lines, collapse = "\n"), "\n")
cat("\nAll GxE outputs were written to:\n",
    normalizePath(GXE_RESULTS_DIR), "\n")


