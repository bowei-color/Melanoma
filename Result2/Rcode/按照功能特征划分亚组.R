setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result2/code/")



rm(list = ls())

library(Seurat)
library(pheatmap)
library(grid)
library(gridExtra)
library(gtable)


# seurat_obj <- readRDS("../data/source/melanocyte_subclusters.rds")
# 
# 
# seurat_obj$subgroup <- factor(
#   ifelse(seurat_obj$seurat_clusters %in% c(2, 9, 10, 14, 15), "Subgroup1",
#          ifelse(seurat_obj$seurat_clusters %in% c(6, 8, 11, 16, 17), "Subgroup2",
#                 ifelse(seurat_obj$seurat_clusters %in% c(0, 1, 3, 18), "Subgroup3",
#                 ifelse(seurat_obj$seurat_clusters %in% c(4, 5, 19), "Subgroup4",
#                        ifelse(seurat_obj$seurat_clusters %in% c(7, 12, 13), "Others", "Others")))))
# )
# 
# saveRDS(seurat_obj, file = "../data/source/melanocyte_subgroup_clusters.rds")


# pathway_to_signature <- list(
#   Signature1 = c("peptide antigen assembly with MHC class II protein complex", "MHC class II protein complex assembly", "antigen processing and presentation of exogenous peptide antigen via MHC class II"),
#   Signature2 = c("heterotypic cell-cell adhesion", "eosinophil migration", "dendritic cell migration", "positive regulation of cell-substrate adhesion"),
#   Signature3 = c("DNA unwinding involved in DNA replication", "positive regulation of chromosome segregation", "regulation of DNA-templated DNA replication initiation", "DNA strand elongation involved in DNA replication"),
#   Signature4 = c("protein refolding", "chaperone cofactor-dependent protein refolding", "response to heat")
# )

# pathway_to_signature <- list(
#   Signature1 = c("peptide antigen assembly with MHC class II protein complex", "MHC class II protein complex assembly", "MHC class II antigen presentation"),
#   Signature2 = c("DNA unwinding involved in DNA replication", "positive regulation of chromosome segregation", "regulation of DNA-templated DNA replication initiation", "DNA strand elongation involved in DNA replication"),
#   Signature3 = c("protein refolding", "chaperone cofactor-dependent protein refolding", "response to heat"),
#   Signature4 = c("heterotypic cell-cell adhesion")
# )




heatmap_matrix_df <- read.csv("../data/result/ssgsea_cluster_heatmap_matrix1.csv", row.names = 1)
colnames(heatmap_matrix_df) <- gsub("^X", "C", colnames(heatmap_matrix_df))

gsva_matrix1 <- as.matrix(heatmap_matrix_df)




df <- read.csv("../data/source/代表性功能通路_用于ssGSEA.csv", stringsAsFactors = FALSE)

# 映射 Functional_Category 到 Signature 名称
category_to_signature <- c(
  Immune_Response = "Signature1",
  Cell_Cycle = "Signature2",
  Translation_Ribosome = "Signature3",
  Stress_Response = "Signature4"
)

# 添加 Signature 字段
df$Signature <- category_to_signature[df$Functional_Category]

# 按 Signature 分组构建 pathway_to_signature 列表
pathway_to_signature <- split(df$Description, df$Signature)



signature <- sapply(rownames(gsva_matrix1), function(pathway) {
  for (sig in names(pathway_to_signature)) {
    if (pathway %in% pathway_to_signature[[sig]]) {
      return(sig)
    }
  }
  return("Others")
})

annotation_row <- data.frame(Signature = signature)
rownames(annotation_row) <- rownames(gsva_matrix1)

cluster_subgroup_df <- data.frame(
  cluster = c("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19"),
  subgroup = c("Subgroup3", "Subgroup3", "Subgroup1", "Subgroup3", "Subgroup4", "Subgroup4", "Subgroup2", "Others", "Subgroup2", "Subgroup1", "Subgroup1", "Subgroup2", "Others", "Others", "Subgroup1", "Subgroup1", "Subgroup2", "Subgroup2", "Subgroup3", "Subgroup4")
)

annotation_col <- data.frame(subgroup = cluster_subgroup_df$subgroup)
rownames(annotation_col) <- cluster_subgroup_df$cluster

annotation_colors <- list(
  Signature = c(
    Signature1 = "#cd161d",
    Signature2 = "#ff840e",
    Signature3 = "#1e6bae",
    Signature4 = "#2ca031"
  ),
  subgroup = c(
    "Subgroup1" = "#cd161d",
    "Subgroup2" = "#ff840e",
    "Subgroup3" = "#1e6bae",
    "Subgroup4" = "#2ca031",
    "Others" = "grey"
  )
)


colors <- colorRampPalette(c("blue", "white", "red"))(100)



pdf("../figure/每个功能每个亚组的ssGSEA评分.pdf", width = 14, height = 6)


pheatmap(gsva_matrix1,
         show_colnames = TRUE,
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         scale = "row",
         annotation_row = annotation_row,
         annotation_col = annotation_col,
         color = colors,
         cellwidth = 20,
         cellheight = 20,
         fontsize = 10,
         labels_row = rownames(gsva_matrix1),
         annotation_names_row = FALSE,
         gaps_row = cumsum(table(signature)),
         display_numbers = TRUE,
         border_color = "white",
         # angle_col = NULL,   # 设置为NULL
         annotation_colors = annotation_colors,
         legend = TRUE,
         fontsize_row = 12,
         fontsize_col = 12,
         annotation_legend = FALSE,
         annotation_names_col = FALSE)




while(dev.cur() > 1) dev.off()

