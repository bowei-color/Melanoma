setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result2/code/")


rm(list = ls())

library(Seurat)
library(dplyr)


seurat_obj <- readRDS("../data/source/melanocyte_subgroup_clusters.rds")



# 计算每个样本中每个细胞类型的数量和百分比
celltype_percentage <- seurat_obj@meta.data %>%
  group_by(sample, subgroup, group) %>%  
  summarise(count = n()) %>%
  group_by(sample) %>%
  mutate(percentage = count / sum(count) * 100) %>%  # 计算百分比
  ungroup()

# 转换sample列为字符型
celltype_percentage$sample <- as.character(celltype_percentage$sample)


# 保存结果到 CSV 文件
write.csv(celltype_percentage, "../data/result/各样本中各黑色素细胞功能特征占比.csv", row.names = FALSE)


