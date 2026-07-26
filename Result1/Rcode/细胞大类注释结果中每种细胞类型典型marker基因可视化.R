
setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/code")


rm(list = ls())

library(Seurat)
library(ggplot2)
library(dplyr)


seurat_obj <- readRDS("E:/工作/论文/黑色素瘤/数据/RDS/大类注释/rename_seuratobj-大类注释.rds")

# DimPlot(seurat_obj, reduction = "umap", group.by = "seurat_clusters", label = TRUE, pt.size = 1) +
#   ggtitle("细胞簇（未注释）可视化") +
#   theme(plot.title = element_text(hjust = 0.5))


#Tcells
ptcells <- FeaturePlot(seurat_obj, features = c("CD3D","CD3E","CD8A","PTPRC","CD4","CD2"))  # TCells
ggsave(filename = "../figure/T细胞典型marker基因表达量可视化图.pdf", plot = ptcells, device = cairo_pdf, width = 12, height = 8, dpi = 600, units = "in")



# Bcells
pbcells <- FeaturePlot(seurat_obj, features = c("PXK", "MS4A1", "CD19", "CD74", "CD79A", "IGHD", "CD79B", "IGHM", "BANK1", "PTPRC", "CD38", "CD27", "CD5", "ICHM", "CD22")) 
ggsave(filename = "../figure/B细胞典型marker基因表达量可视化图.pdf", plot = pbcells, device = cairo_pdf, width = 12, height = 8, dpi = 600, units = "in" )


#MPs细胞
pmpscells <- FeaturePlot(seurat_obj, features = c("CD68", "LYZ", "CD14", "CD163", "MRC1", "MS4A7"))
ggsave(filename = "../figure/MPs细胞典型marker基因表达量可视化图.pdf", plot = pmpscells, device = cairo_pdf, width = 12, height = 8, dpi = 600, units = "in")


# Fibroblasts
pfibroblasts <- FeaturePlot(seurat_obj, features = c("VIM", "PDGFRB", "LUM", "COL6A2", "VTN", "MFAP5", "COL1A2", "COL1A1", "SERPINH1", "POSTN", "ASPN", "PRRX1", "COL6A3", "PDGFRA")) 
ggsave(filename = "../figure/成纤维细胞(Fibroblasts)典型marker基因表达量可视化图.pdf", plot = pfibroblasts, device = cairo_pdf, width = 12, height = 8, dpi = 600, units = "in")


# Mast cells
pmast <- FeaturePlot(seurat_obj, features = c("HSD11B1", "SLC29A1", "LUM", "COL6A2", "VTN", "MFAP5", "COL1A2", "COL1A1", "SERPINH1", "POSTN", "ASPN", "PRRX1", "COL6A3", "PDGFRA")) 


# Keratinocytes
FeaturePlot(seurat_obj, features = c("KRT14", "KRT1", "KRT10"))


#Tcells
FeaturePlot(seurat_obj, features = c("CD3D", "CD2", "TRBC2"))
FeaturePlot(seurat_obj, features = c("IL7R", "CD3E", "CD3G"))


#Bcells
FeaturePlot(seurat_obj, features = c("PAX5", "MS4A1"))

FeaturePlot(seurat_obj, features = c("PXK"))



# Mononuclear phagocytes
FeaturePlot(seurat_obj, features = c("LYZ", "CD14"))



# Melanocytes:
FeaturePlot(seurat_obj, features = c("MLANA", "PMEL", "DCT", "TYRP1"))
FeaturePlot(seurat_obj, features = c("MITF", "RAB38", "PRKN", "CCN5"))
FeaturePlot(seurat_obj, features = c("MLANA", "PMEL", "DCT", "MITF"))

# Fibroblasts
FeaturePlot(seurat_obj, features = c("DCN", "LUM", "COL1A1"))


# Lymphatic endothelial cells
FeaturePlot(seurat_obj, features = c("CCL21", "PROX1"))


# Vascular endothelial cells
FeaturePlot(seurat_obj, features = c("CDH5", "PECAM1", "PLVAP"))
FeaturePlot(seurat_obj, features = c("CD93", "VWF", "EMCN"))


# Mast Cells
FeaturePlot(seurat_obj, features = c("TPSB2", "CPA3"))
FeaturePlot(seurat_obj, features = c("KIT", "HSD11B1"))


# Smooth muscle cells
FeaturePlot(seurat_obj, features = c("MYH11", "ACTA2"))



# Schwann cells
FeaturePlot(seurat_obj, features = c("MPZ", "SCN7A", "CDH19"))
FeaturePlot(seurat_obj, features = c("PLP1"))



# Adipocytes
FeaturePlot(seurat_obj, features = c("ADIPOQ", "PLIN1", "FABP4"))




