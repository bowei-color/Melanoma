# 
# rm(list = ls())
# 
# 
# library(Seurat)
# 
# 
# seurat_merged <- readRDS("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/data/result/seurat_integrated.rds")
# 
# 
# 
# # 降维聚类
# DefaultAssay(seurat_merged) <- "integrated"
# seurat_merged <- ScaleData(seurat_merged, verbose = FALSE)
# seurat_merged <- RunPCA(seurat_merged, npcs = 30, verbose = FALSE)
# seurat_merged <- RunUMAP(seurat_merged, dims = 1:30)
# seurat_merged <- FindNeighbors(seurat_merged, dims = 1:30)
# seurat_merged <- FindClusters(seurat_merged, resolution = 0.5)
# 
# # 可视化
# DimPlot(seurat_merged, group.by = "group", label = TRUE)   # AM vs CM
# DimPlot(seurat_merged, group.by = "sample", label = TRUE)  # 每个样本http://127.0.0.1:11041/graphics/fbac413d-1f21-48e6-914d-313085ffd138.png
# DimPlot(seurat_merged, group.by = "seurat_clusters", label = TRUE)
# 
# 
# DefaultAssay(seurat_merged) <- "RNA"
# FeaturePlot(seurat_merged, features = c("CD3D", "CD2", "TRBC2")) # T cells  2 3 9 24 12
# 
# FeaturePlot(seurat_merged, features = c("PAX5", "MS4A1")) # B Cells 
# 
# table(seurat_merged$group)

rm(list = ls())

library(Seurat)
library(ggplot2)
library(dplyr)
library(readr)
library(tidyr)
library(stringr)



# 自定义颜色
celltype_colors <- c(
  "TCells"        = "#e44a34",
  "BCells"        = "#48b9d3",
  "MPs"           = "#1f9f85",
  "Keratinocytes" = "#375287",
  "Melanocytes"   = "#ef9a7e",
  "Fibroblasts"   = "#8490b3",
  "LECs"          = "#92cebf",
  "VascularECs"   = "#db0a17",
  "MastCells"     = "#7c5f45",
  "SGCs"          = "#cccccc",
  "SMCs"          = "#ba8a83",
  "SchwannCells"  = "#24abac",
  "Adipocytes"    = "#2e7987"
)


seurat_merged <- readRDS("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/data/result/seurat_integrated.rds")



mapping_df <- read_csv("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/data/source/新的细胞类型与簇对应关系.csv")
mapping_long <- mapping_df %>%
  mutate(cluster = str_remove_all(cluster, '\\\\"')) %>%        # 移除转义引号 \" 
  mutate(cluster = str_remove_all(cluster, '"')) %>%            # 再移除普通引号 "
  separate_rows(cluster, sep = ",") %>%                         # 拆分为多行
  mutate(cluster = str_trim(cluster)) %>%                       # 去除多余空格
  mutate(cluster = as.character(as.integer(cluster)))           # 变为字符编号


DimPlot(seurat_merged, reduction = "umap", group.by = "seurat_clusters", label = TRUE)
DimPlot(seurat_merged, reduction = "umap", group.by = "group")
DimPlot(seurat_merged, reduction = "umap", group.by = "sample", label = TRUE)

Idents(seurat_merged) <- "seurat_clusters"
seurat_merged$celltype <- mapping_long$cell_type[match(as.character(Idents(seurat_merged)), mapping_long$cluster)]

# 保存注释结果
saveRDS(seurat_merged, file = "E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/data/result/seurat_annotated_obj.rds")



# UMAP 可视化
DimPlot(seurat_merged, group.by = "celltype", label = FALSE)


umap_df <- as.data.frame(Embeddings(seurat_merged, "umap"))
colnames(umap_df) <- c("UMAP_1", "UMAP_2")
umap_df$celltype <- seurat_merged$celltype

p <- ggplot(umap_df, aes(x = UMAP_1, y = UMAP_2, color = celltype)) +
  geom_point(size = 0.5, alpha = 0.8) +
  scale_color_manual(values = celltype_colors) +
  labs(title = "", x = "UMAP_1", y = "UMAP_2") +
  guides(color = guide_legend(override.aes = list(size = 4))) +
  theme_minimal(base_size = 14) +
  theme(
    axis.line = element_line(color = "black", size = 0.8),
    axis.ticks = element_line(color = "black"),
    axis.text = element_text(size = 14, color = "black"),
    axis.title = element_text(size = 14),
    panel.grid = element_blank(),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14),
    legend.key = element_blank()
  )
ggsave("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/figure/AM和CM的Umap图.pdf", p, width = 8, height = 8, dpi = 600)


# 保存画UMAP图所需数据

umap_reslut <- as.data.frame(Embeddings(seurat_merged, "umap"))
colnames(umap_reslut) <- c("UMAP_1", "UMAP_2")


umap_reslut$celltype <- seurat_merged$celltype
umap_reslut$cluster <- Idents(seurat_merged)
umap_reslut$group <- seurat_merged$group
umap_reslut$sample <- seurat_merged$sample

# 保存为 CSV 文件
write.csv(umap_reslut, "E:/工作/论文/黑色素瘤/code/MelanomaAMCM/MelanomaAMCM/data/result1/source/画肢端和皮肤黑色素瘤UMAP的数据.csv", row.names = FALSE)





table(seurat_merged$group)

# marker基因表达可视化

# FeaturePlot(seurat_merged, features = c("CD3D", "CD3E", "TRBC2")) # T cells 3  6 10 11
# FeaturePlot(seurat_merged, features = c("CD79A", "MS4A1")) # B Cells 12
# FeaturePlot(seurat_merged, features = c("LYZ", "CD14", "CD68")) # MPs(monocytes and macrophages)  7 16
# FeaturePlot(seurat_merged, features = c("KRT14", "KRT1")) # Keratinocytes 8 14 29
# FeaturePlot(seurat_merged, features = c("MLANA","PMEL","DCT","TYRP1")) # Melanocytes 2 4 5 19 24 25
# FeaturePlot(seurat_merged, features = c("DCN","LUM","COL1A1")) # Fibroblasts    1 23 26 27
# FeaturePlot(seurat_merged, features = c("CCL21","PROX1")) # LECs 21
# FeaturePlot(seurat_merged, features = c("CDH5","PECAM1","PLVAP")) # VascularECs 9 18
# FeaturePlot(seurat_merged, features = c("TPSB2","CPA3")) # MastCells 28
# FeaturePlot(seurat_merged, features = c("DCD","SAA1","LTF")) # SGCs(Sweat gland cells) 15 22
# FeaturePlot(seurat_merged, features = c("MYH11","ACTA2")) # SMCs(Smooth muscle cells)   13 
# FeaturePlot(seurat_merged, features = c("MPZ","CDH19")) # SchwannCells # 20
# FeaturePlot(seurat_merged, features = c("ADIPOQ","PLIN1","FABP4")) # Adipocytes 17
