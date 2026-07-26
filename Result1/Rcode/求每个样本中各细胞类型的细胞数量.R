
setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/code/")


rm(list = ls())

library(Seurat)


seurat_obj <- readRDS("../data/result/seurat_annotated_obj.rds")

head(seurat_obj)


meta_data <- seurat_obj@meta.data
celltype_counts <- meta_data %>%
  group_by(sample = sample, celltype = celltype) %>%  
  summarise(count = n(), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = celltype, values_from = count, values_fill = 0)
write.csv(celltype_counts, file = "../data/result/每个样本中各细胞类型的细胞数量.csv", row.names = FALSE)
