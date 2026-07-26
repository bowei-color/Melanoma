setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result2/code/")


rm(list = ls())


library(Seurat)
library(ggplot2)


seurat_obj <- readRDS("../data/source/melanocyte_subgroup_clusters.rds")

tsne_coord <- Embeddings(seurat_obj, reduction = "tsne")



meta_info <- seurat_obj@meta.data[, c("seurat_clusters", "sample", "group", "subgroup")]

tsne_df <- cbind(tsne_coord, meta_info[rownames(tsne_coord), ])

write.csv(tsne_df, file = "../data/result/黑色素细胞亚群细分tsne数据.csv", row.names = TRUE)


# DimPlot(seurat_obj, reduction = "tsne", group.by = "seurat_clusters", pt.size = 1, label = TRUE) +
#   labs(color = "cluster") +  
#   theme(
#     axis.text = element_text(size = 18),
#     axis.title = element_text(size = 18),
#     axis.title.x = element_text(margin = margin(t = 10)),
#     axis.line = element_line(size = 0.7, color = "black"),
#     axis.ticks = element_line(size = 0.7),
#     axis.ticks.length = unit(8, "pt"),
#     legend.text = element_text(size = 20),
#     legend.title = element_text(size = 20)
#   ) +
#   ggtitle("")
