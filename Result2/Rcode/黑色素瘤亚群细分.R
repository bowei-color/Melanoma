setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result2/code/")


rm(list = ls())


library(Seurat)

seurat_obj <- readRDS("../../result1/data/result/seurat_annotated_obj.rds")

# head(seurat_obj)


melano_obj <- subset(seurat_obj, subset = celltype == "Melanocytes")


raw_counts <- GetAssayData(melano_obj, assay = "RNA", layer = "counts")
melano_obj[["RNA"]] <- CreateAssayObject(counts = raw_counts)
DefaultAssay(melano_obj) <- "RNA"
melano_obj <- NormalizeData(melano_obj)
melano_obj <- FindVariableFeatures(melano_obj, selection.method = "vst", nfeatures = 2000)
melano_obj <- ScaleData(melano_obj)
melano_obj <- RunPCA(melano_obj)



melano_obj <- FindNeighbors(melano_obj, dims = 1:20)
melano_obj <- FindClusters(melano_obj, resolution = 0.5)
melano_obj <- RunUMAP(melano_obj, dims = 1:20)
melano_obj <- RunTSNE(melano_obj, dims = 1:20)

DimPlot(melano_obj, reduction = "umap", group.by = "seurat_clusters", label = TRUE)
DimPlot(melano_obj, reduction = "tsne", group.by = "seurat_clusters", pt.size = 0.5, label = TRUE)


markers <- FindAllMarkers(melano_obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)

saveRDS(melano_obj, file = "../data/source/melanocyte_subclusters.rds")
write.csv(markers, file = "../data/source/melanocyte_subcluster_markers.csv", row.names = FALSE)
