
setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result2/code/")


rm(list = ls())

library(Seurat)
library(GSVA)
library(clusterProfiler)
library(org.Hs.eg.db)
library(pheatmap)
library(dplyr)

go_data <- read.csv("../data/result/GO显著通路汇总_p小于0.01.csv")




df <- read.csv("../data/source/代表性功能通路_用于ssGSEA.csv", stringsAsFactors = FALSE)

df$Signature <- dplyr::recode(df$Functional_Category,
                              Immune_Response = "Signature1",
                              Cell_Cycle = "Signature2",
                              Translation_Ribosome = "Signature3",
                              Stress_Response = "Signature4")

Signature1 <- df$Description[df$Signature == "Signature1"]
Signature2 <- df$Description[df$Signature == "Signature2"]
Signature3 <- df$Description[df$Signature == "Signature3"]
Signature4 <- df$Description[df$Signature == "Signature4"]


gene_modules <- list()

for (signature_name in list(Signature1, Signature2, Signature3, Signature4)) {
  signature_genes <- c()
  for (term in signature_name) {
    go_result <- go_data %>% filter(grepl(term, Description))
    if (nrow(go_result) > 0) {
      genes <- go_result$geneID
      signature_genes <- unique(c(signature_genes, unlist(strsplit(genes, "/"))))
    }
  }
  signature_list <- list(Signature1, Signature2, Signature3, Signature4)
  signature_index <- which(sapply(signature_list, function(x) identical(x, signature_name)))
  gene_modules[[paste("Signature", signature_index, sep="")]] <- signature_genes
}

seurat_obj <- readRDS("../data/source/melanocyte_subgroup_clusters.rds")
expr_matrix <- GetAssayData(seurat_obj, slot = "data")


ssgsea_scores_matrix <- data.frame(matrix(ncol = length(gene_modules), nrow = length(unique(seurat_obj$sample))))
colnames(ssgsea_scores_matrix) <- names(gene_modules)
rownames(ssgsea_scores_matrix) <- unique(seurat_obj$sample)


for (sample in unique(seurat_obj@meta.data$sample)) {
  sample_cells <- rownames(seurat_obj@meta.data)[seurat_obj@meta.data$sample == sample]
  sample_expr_matrix <- expr_matrix[, sample_cells]
  
  gsvaPar <- ssgseaParam(
    exprData = sample_expr_matrix,
    geneSets = gene_modules,
    normalize = TRUE
  )
  
  ssgsea_scores <- gsva(gsvaPar, verbose = FALSE)
  avg_scores <- rowMeans(ssgsea_scores, na.rm = TRUE)
  
  ssgsea_scores_matrix[sample, ] <- avg_scores
}



rownames(ssgsea_scores_matrix) <- sub(".*_", "", rownames(ssgsea_scores_matrix))

normalized_scores_matrix <- t(scale(t(ssgsea_scores_matrix)))

# write.csv(normalized_scores_matrix, "E:/工作/论文/黑色素瘤/code/Melanoma/data/Result2/黑色素细胞每个样本的ssGSEA功能评分.csv", row.names = TRUE)
# write.csv(ssgsea_scores_matrix, "E:/工作/论文/黑色素瘤/code/Melanoma/data/Result2/黑色素细胞每个样本的ssGSEA功能评分（未标准化）.csv", row.names = TRUE)


# scores_matrix_read <- read.csv("E:/工作/论文/黑色素瘤/code/Melanoma/data/Result2/黑色素细胞每个样本的ssGSEA功能评分（未标准化）.csv", row.names = 1)


# cluster_annotation <- data.frame(
#   Cluster = ifelse(rownames(scores_matrix_read) %in% c("SHH2", "ZTY1", "WBZ2", "FGQ1", "FGQ2", "HYM1", "SL1"), "C1", "C2"),
#   row.names  = rownames(scores_matrix_read)
# )

# annotation_colors <- list(
#   Cluster = c(C1 = "#E41A1C", C2 = "#377EB8")  # 红蓝配色
# )

# pdf("../figure/黑色素细胞各样本按照功能聚类分组.pdf", width = 8, height = 6)



pheatmap(normalized_scores_matrix,
         cluster_rows = TRUE,
         cluster_cols = FALSE,
         scale = "row",
         show_rownames = TRUE,
         show_colnames = TRUE,
         color = colorRampPalette(c("#496e9f", "white", "#be382d"))(100),
         # annotation_row = cluster_annotation,
         # annotation_colors = annotation_colors,
         annotation_names_row = FALSE) 

# dev.off()
