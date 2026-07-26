# setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result2/code/")
# 
# rm(list = ls())
# 
# library(Seurat)
# library(GSVA)
# library(pheatmap)
# 
# 
# 
# seurat_obj <- readRDS("../data/source/melanocyte_subclusters.rds")
# markers <- read.csv("../data/source/melanocyte_subcluster_markers.csv")
# 
# 
# if (length(markers$gene) == 0) {
#   stop("No marker genes found. Please check the output of FindAllMarkers.")
# }
# 
# 
# selected_go_terms <- c(
#   "peptide antigen assembly with MHC class II protein complex",          # Immune_Response
#   "MHC class II protein complex assembly",
#   "antigen processing and presentation of exogenous peptide antigen via MHC class II",
#   
#   "DNA unwinding involved in DNA replication",                           # CellCycle
#   "positive regulation of chromosome segregation",
#   "regulation of DNA-templated DNA replication initiation",
#   "DNA strand elongation involved in DNA replication",
#   
#   "protein refolding",                                                  # Stress_Response
#   "chaperone cofactor-dependent protein refolding",
#   "response to heat",
#   
#   "heterotypic cell-cell adhesion"                                    # Microenvironment_Remodeling
# 
# )
# 
# # 构建基因集，选择每个GO通路相关的基因
# gene_modules <- list()
# for (term in selected_go_terms) {
#   go_result <- enrichGO(gene = markers$gene,
#                         OrgDb = org.Hs.eg.db,
#                         keyType = "SYMBOL",
#                         ont = "BP",
#                         pvalueCutoff = 0.05,
#                         qvalueCutoff = 0.2,
#                         readable = TRUE)
#   
#   if (!is.null(go_result)) {
#     genes <- go_result@result %>%
#       filter(Description == term) %>%
#       pull(geneID)  
#     gene_modules[[term]] <- unlist(strsplit(genes, "/")) 
#   }
# }
# 
# # 提取表达矩阵
# expr_matrix <- GetAssayData(seurat_obj, slot = "data")
# 
# # 为每个簇计算 ssGSEA 打分
# cluster_ssgsea_scores <- list()
# for (cl in unique(seurat_obj$seurat_clusters)) {
#   cluster_cells <- WhichCells(seurat_obj, idents = cl)
#   cluster_expr_matrix <- expr_matrix[, cluster_cells]  # 提取该簇对应的细胞表达矩阵
#   
#   # 创建ssGSEA参数对象
#   gsvaPar <- ssgseaParam(
#     exprData = cluster_expr_matrix,
#     geneSets = gene_modules,
#     normalize = TRUE
#   )
#   
#   # 计算 ssGSEA 打分
#   ssgsea_scores <- gsva(gsvaPar, verbose = FALSE)
#   
#   # 保存每个cluster的ssGSEA得分
#   cluster_ssgsea_scores[[as.character(cl)]] <- ssgsea_scores
# }
# 
# 
# 
# # 获取挑选出的GO通路
# all_go_terms <- selected_go_terms
# 
# # 初始化空矩阵存储每个簇的平均ssGSEA评分
# heatmap_matrix <- matrix(NA, nrow = length(all_go_terms), ncol = length(cluster_ssgsea_scores))
# 
# # 计算每个簇的平均 ssGSEA 得分，并填充矩阵
# for (i in seq_along(cluster_ssgsea_scores)) {
#   cluster_name <- names(cluster_ssgsea_scores)[i]
#   cluster_score <- cluster_ssgsea_scores[[cluster_name]]
#   
#   # 获取当前簇的 GO 通路顺序（行名）
#   current_go_terms <- rownames(cluster_score)
#   
#   # 只填充与所有 GO 通路集合重合的部分
#   common_go_terms <- intersect(all_go_terms, current_go_terms)
#   
#   if (length(common_go_terms) > 0) {
#     # 计算每个 GO 通路的平均得分
#     avg_scores <- rowMeans(cluster_score[common_go_terms, , drop = FALSE], na.rm = TRUE)
#     
#     # 确保顺序一致，并填充矩阵
#     heatmap_matrix[match(common_go_terms, all_go_terms), i] <- avg_scores
#   }
# }
# 
# 
# # 检查矩阵是否有 NA 或零值，并填充
# heatmap_matrix[is.na(heatmap_matrix)] <- 0
# 
# # 设置列名和行名
# colnames(heatmap_matrix) <- names(cluster_ssgsea_scores)
# rownames(heatmap_matrix) <- all_go_terms
# 
# # 绘制热图
# pheatmap(heatmap_matrix,
#          cluster_rows = FALSE,  # 禁用行聚类
#          cluster_cols = FALSE,  # 禁用列聚类
#          scale = "row",         # 对行进行标准化
#          show_rownames = TRUE,  # 显示行名
#          show_colnames = TRUE,  # 显示列名
#          color = colorRampPalette(c("blue", "white", "red"))(100),  # 配色
#          main = "ssGSEA Score Heatmap for Clusters")  # 标题
# 
#  # 将 heatmap_matrix 转换为 data.frame
# # heatmap_matrix_df <- as.data.frame(heatmap_matrix)
# 
# 
# 
# # 保存为 CSV 文件
# # write.csv(heatmap_matrix_df, "../data/result/ssgsea_cluster_heatmap_matrix.csv", row.names = TRUE)


setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result2/code/")

rm(list = ls())

library(Seurat)
library(GSVA)
library(pheatmap)
library(dplyr)


seurat_obj <- readRDS("../data/source/melanocyte_subclusters.rds")
gene_set_df <- read.csv("../data/source/代表性功能通路_用于ssGSEA.csv", stringsAsFactors = FALSE)

# 构建 gene_modules：每个GO term 对应一个基因集
gene_modules <- list()
for (term in unique(gene_set_df$Description)) {
  genes <- gene_set_df %>% filter(Description == term) %>% pull(geneID)
  gene_modules[[term]] <- unique(unlist(strsplit(genes, "/")))
}

# 提取表达矩阵
expr_matrix <- GetAssayData(seurat_obj, slot = "data")

# 为每个簇计算 ssGSEA 打分
cluster_ssgsea_scores <- list()
for (cl in unique(seurat_obj$seurat_clusters)) {
  cluster_cells <- WhichCells(seurat_obj, idents = cl)
  cluster_expr_matrix <- expr_matrix[, cluster_cells]
  
  gsvaPar <- ssgseaParam(
    exprData = cluster_expr_matrix,
    geneSets = gene_modules,
    normalize = TRUE
  )
  
  ssgsea_scores <- gsva(gsvaPar, verbose = FALSE)
  cluster_ssgsea_scores[[as.character(cl)]] <- ssgsea_scores
}

# 构建通路×cluster打分矩阵
all_go_terms <- names(gene_modules)
heatmap_matrix <- matrix(NA, nrow = length(all_go_terms), ncol = length(cluster_ssgsea_scores))

for (i in seq_along(cluster_ssgsea_scores)) {
  cluster_name <- names(cluster_ssgsea_scores)[i]
  cluster_score <- cluster_ssgsea_scores[[cluster_name]]
  current_go_terms <- rownames(cluster_score)
  common_go_terms <- intersect(all_go_terms, current_go_terms)
  
  if (length(common_go_terms) > 0) {
    avg_scores <- rowMeans(cluster_score[common_go_terms, , drop = FALSE], na.rm = TRUE)
    heatmap_matrix[match(common_go_terms, all_go_terms), i] <- avg_scores
  }
}

heatmap_matrix[is.na(heatmap_matrix)] <- 0
colnames(heatmap_matrix) <- names(cluster_ssgsea_scores)
rownames(heatmap_matrix) <- all_go_terms

# 绘制热图
pheatmap(heatmap_matrix,
         cluster_rows = FALSE,
         cluster_cols = TRUE,
         scale = "row",
         show_rownames = TRUE,
         show_colnames = TRUE,
         color = colorRampPalette(c("blue", "white", "red"))(100),
         main = "ssGSEA Score Heatmap for Clusters")



 # 将 heatmap_matrix 转换为 data.frame
# heatmap_matrix_df <- as.data.frame(heatmap_matrix)



# 保存为 CSV 文件
# write.csv(heatmap_matrix_df, "../data/result/ssgsea_cluster_heatmap_matrix.csv", row.names = TRUE)
