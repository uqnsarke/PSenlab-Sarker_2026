#!/usr/bin/env Rscript
# ==============================================================================
# Export per-nucleus ATAC QC metrics (Extended Data Fig. 1d)
# ==============================================================================
#
# Description:
#   Writes one row per nucleus with the sample, age group, sex and the two ATAC
#   quality metrics shown in Extended Data Fig. 1d:
#     - nFrags           : ATAC fragments per nucleus (ArchR cellColData)
#     - n_features_ATAC  : number of peaks with non-zero counts per nucleus
#                          (PeakMatrix)
#   The table is read by 10_ED_Fig1d_atac_qc_metrics_statistics.py.
#
# Prerequisites:
#   - Steps 1-7 (01_step1_Arrow creation.R, 02_step2-7_preprocessing.R)
#   - An ArchR project that contains all nuclei and a PeakMatrix
#     (addPeakMatrix, see 06_step11_peak calling and linkage.R)
#   - cellColData columns: Sample, age, sex (Sample is set by ArchR)
#
# Input:
#   - ArchR project with all nuclei and a PeakMatrix
#
# Output:
#   - atac_qc_per_nucleus.csv (row names = ArchR cell names)
#
# Requirements:
#   - R >= 4.0
#   - ArchR >= 1.0.2
#
# ==============================================================================

suppressPackageStartupMessages({
    library(ArchR)
    library(Matrix)
})

# Define Paths (modify according to your directory structure)
PROJ_PATH  <- "path/to/ArchR_Projects/project_with_PeakMatrix"
OUTPUT_CSV <- "qc_metrics/atac_qc_per_nucleus.csv"

dir.create(dirname(OUTPUT_CSV), showWarnings = FALSE, recursive = TRUE)

proj <- loadArchRProject(path = PROJ_PATH)

# Fragments per nucleus plus the sample / age / sex labels
cd <- as.data.frame(getCellColData(proj, select = c("Sample", "age", "sex", "nFrags")))

# Features per nucleus = number of peaks with non-zero counts
pm  <- getMatrixFromProject(proj, useMatrix = "PeakMatrix")
mat <- SummarizedExperiment::assay(pm)
cd$n_features_ATAC <- Matrix::colSums(mat > 0)[rownames(cd)]

stopifnot(!anyNA(cd$n_features_ATAC))
write.csv(cd, OUTPUT_CSV, row.names = TRUE)
message(sprintf("Exported %d nuclei, %d samples: %s",
                nrow(cd), length(unique(cd$Sample)), OUTPUT_CSV))
