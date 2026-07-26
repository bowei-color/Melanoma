setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/code/")


rm(list = ls())

library(Seurat)

seurat_obj <- readRDS("../data/result/seurat_annotated_obj.rds")

head(seurat_obj)

meta_data <- seurat_obj@meta.data
celltype_counts <- meta_data %>%
  group_by(sample = celltype, celltype = group) %>%
  summarise(count = n(), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = celltype, values_from = count, values_fill = 0)

write.csv(celltype_counts, file = "../data/result/每种细胞类型中每个分组的细胞数量.csv", row.names = FALSE)


cat("总细胞数（列）:", ncol(seurat_obj), "\n")
cat("总特征数（行）:", nrow(seurat_obj), "\n")

table(seurat_obj$group)



