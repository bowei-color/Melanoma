setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result3/code/")



rm(list = ls())


library(pheatmap)


heatmap_matrix_df <- read.csv("../data/result/Adjusted_ssGSEA_Matrix.csv", row.names = 1)
heatmap_matrix_df <- heatmap_matrix_df
gsva_matrix1 <- as.matrix(heatmap_matrix_df)


df <- read.csv("../../result2/data/source/代表性功能通路_用于ssGSEA.csv", stringsAsFactors = FALSE)


category_to_signature <- c(
  Immune_Response = "Signature1",
  Cell_Cycle = "Signature2",
  Translation_Ribosome = "Signature3",
  Stress_Response = "Signature4"
)


df$Signature <- category_to_signature[df$Functional_Category]

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

sample_label_map <- read.csv("../data/result/bulk数据打分后样本和标签的对应关系.csv", row.names = 1)




sample_group_df <- data.frame(
  samplename = rownames(sample_label_map),
  samplegroup = sample_label_map$Group,
  stringsAsFactors = FALSE
)

annotation_col <- data.frame(samplegroup = sample_group_df$samplegroup)
rownames(annotation_col) <- sample_group_df$samplename

annotation_colors <- list(
  Signature = c(
    Signature1 = "#cd161d",
    Signature2 = "#ff840e",
    Signature3 = "#1e6bae",
    Signature4 = "#2ca031"
  ),
  samplegroup = c(
    "AM" = "#cd161d",
    "CM" = "#1e6bae",
    "Others" = "grey"
  )
)

colors <- colorRampPalette(c("blue", "white", "red"))(100)


# max_val <- max(gsva_matrix1, na.rm = TRUE)
# min_val <- min(gsva_matrix1, na.rm = TRUE)
# abs_max <- max(abs(min_val), abs(max_val))
# breaks <- seq(-abs_max, abs_max, length.out = 204)
# colors <- colorRampPalette(c("blue", "white", "red"))(200)



pdf("../figure/bulk数据每个功能的ssGSEA评分.pdf", width = 8, height = 6)

pheatmap(gsva_matrix1,
         show_colnames = FALSE,
         show_rownames = FALSE,
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         # scale = "column",
         annotation_row = annotation_row,
         annotation_col = annotation_col,
         color = colors,
         # breaks = breaks,
         cellwidth = 5,
         cellheight = 20,
         fontsize = 20,
         # labels_row = rownames(gsva_matrix1),
         annotation_names_row = FALSE,
         gaps_row = cumsum(table(signature)),
         display_numbers = FALSE,
         border_color = "white",
         annotation_colors = annotation_colors,
         legend = FALSE,
         fontsize_row = 20,
         fontsize_col = 20,
         annotation_legend = FALSE,
         annotation_names_col = FALSE)

while(dev.cur() > 1) dev.off()



