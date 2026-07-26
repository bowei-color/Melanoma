setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result3/code/")

rm(list = ls())

library(Seurat)
library(dplyr)

seurat_obj <- readRDS("../../result2/data/source/melanocyte_subgroup_clusters.rds")

# 设置身份为 subgroup
Idents(seurat_obj) <- seurat_obj$subgroup

# 获取所有 subgroup 名称
subgroups <- unique(seurat_obj$subgroup)



# 创建列表保存每个 subgroup 的 top 100 marker
top100_markers_list <- list()

# 对每个 subgroup 依次与其他所有细胞进行差异分析
for (sg in subgroups) {
  markers <- FindMarkers(seurat_obj, ident.1 = sg, ident.2 = NULL, only.pos = TRUE)
  top100 <- markers %>%
    arrange(desc(avg_log2FC)) %>%
    head(100)
  top100$gene <- rownames(top100)
  top100$subgroup <- sg
  top100_markers_list[[sg]] <- top100
}

# 合并所有 top10 结果为一个 data frame
top100_all <- bind_rows(top100_markers_list)

# 保存结果
write.csv(top100_all, "../data/result/各subgroup的top100差异基因.csv", row.names = FALSE)


# 每个 subgroup 中选择 avg_log2FC 排名前10的基因
top10_each <- top100_all %>%
  group_by(subgroup) %>%
  arrange(desc(avg_log2FC), .by_group = TRUE) %>%
  slice_head(n = 10) %>%
  ungroup()

# 保存结果
write.csv(top10_each, "../data/result/各subgroup的top10差异基因.csv", row.names = FALSE)



# # 提取表达矩阵（行为基因，列为细胞）
# expr_matrix <- GetAssayData(seurat_obj, assay = "RNA", layer = "data")
# 
# # 提取细胞元信息（确保顺序一致）
# meta_info <- seurat_obj@meta.data[colnames(expr_matrix), c("group", "sample", "subgroup")]
# 
# # 转置表达矩阵（行：细胞，列：基因），并合并元信息
# expr_with_meta <- cbind(meta_info, t(as.matrix(expr_matrix)))
# 
# # 可选保存
# write.csv(expr_with_meta, "../data/result/expression_with_group_sample_subgroup.csv", row.names = TRUE)


