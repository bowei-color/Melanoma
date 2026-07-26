setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result3/code/")

rm(list = ls())


library(GSVA)
library(pheatmap)
library(dplyr)
library(tibble)
library(pheatmap)

raw_data <- read.csv("E:/工作/论文/黑色素瘤/code/MelanomaAMCM/MelanomaAMCM/data/result3/source/GSE190113_final_data.csv", , stringsAsFactors = FALSE)
labels <- raw_data$label
expr_data <- raw_data[, -1]
rownames(expr_data) <- paste0("Sample", seq_len(nrow(expr_data)))

gene_set_df <- read.csv("../../result2/data/source/代表性功能通路_用于ssGSEA.csv", stringsAsFactors = FALSE)
gene_modules <- list()
for (term in unique(gene_set_df$Description)) {
  genes <- gene_set_df %>% filter(Description == term) %>% pull(geneID)
  gene_modules[[term]] <- unique(unlist(strsplit(genes, "/")))
}


expr_matrix <- t(as.matrix(expr_data))
expr_matrix <- expr_matrix[rownames(expr_matrix) %in% unique(unlist(gene_modules)), ]


gsvaPar <- ssgseaParam(exprData = expr_matrix, geneSets = gene_modules, 
                       normalize = TRUE)


ssgsea_scores <- gsva(gsvaPar)

all_go_terms <- names(gene_modules)
scored_go_terms <- rownames(ssgsea_scores)
missing_go_terms <- setdiff(all_go_terms, scored_go_terms)
if (length(missing_go_terms) > 0) {
  zero_matrix <- matrix(0, nrow = length(missing_go_terms), ncol = ncol(ssgsea_scores),
                        dimnames = list(missing_go_terms, colnames(ssgsea_scores)))
  ssgsea_scores <- rbind(ssgsea_scores, zero_matrix)
}
ssgsea_scores <- ssgsea_scores[all_go_terms, , drop = FALSE]

ssgsea_scores_t <- as.data.frame(t(ssgsea_scores))
ssgsea_scores_t$label <- labels
rownames(ssgsea_scores_t) <- rownames(expr_data)

write.csv(ssgsea_scores_t, "../data/result/ssgsea_bulk_scores_with_label.csv", row.names = TRUE)


heatmap_matrix <- ssgsea_scores_t[, setdiff(colnames(ssgsea_scores_t), "label")]
sample_labels <- ssgsea_scores_t$label
names(sample_labels) <- rownames(ssgsea_scores_t)

sorted_samples <- names(sort(sample_labels, decreasing = TRUE))
heatmap_matrix <- heatmap_matrix[sorted_samples, ]
annotation_row <- data.frame(Group = factor(sample_labels[sorted_samples], levels = c(1, 0)))
rownames(annotation_row) <- sorted_samples

heatmap_scaled <- t(heatmap_matrix)
heatmap_scaled <- t(apply(heatmap_scaled, 1, function(x) {
  if (sd(x) == 0) {
    rep(0, length(x))  
  } else {
    (x - mean(x)) / sd(x)
  }
}))


heatmap_scaled[is.na(heatmap_scaled)] <- 0
heatmap_scaled[is.infinite(heatmap_scaled)] <- 0

rownames(heatmap_scaled) <- colnames(heatmap_matrix) 
colnames(heatmap_scaled) <- rownames(heatmap_matrix) 

pheatmap(heatmap_scaled,
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         # annotation_row = annotation_row,
         # scale = "row",
         show_rownames = TRUE,
         show_colnames = TRUE,
         color = colorRampPalette(c("blue", "white", "red"))(100),
         main = "ssGSEA Score Heatmap (Bulk Samples)")



# 保存 heatmap_matrix 为 CSV
heatmap_matrix_df <- as.data.frame(heatmap_scaled)



label_table <- data.frame(
  Sample = sorted_samples,
  label = sample_labels[sorted_samples]
)

label_table$Group <- ifelse(label_table$label == 1, "AM", "CM")

write.csv(heatmap_matrix_df, "../data/result/ssgsea_bulk_heatmap_matrix.csv", row.names = TRUE)
write.csv(label_table, "../data/result/bulk数据打分后样本和标签的对应关系.csv", row.names = FALSE)
