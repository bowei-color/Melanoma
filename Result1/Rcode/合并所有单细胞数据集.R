
# rm(list = ls())
# 
# 
# library(Seurat)
# library(dplyr)
# 
# 
# path_am <- "E:/工作/论文/黑色素瘤/数据/现有结果黑素瘤/cellranger/"
# path_cm_zty1 <- "E:/工作/论文/黑色素瘤/数据/现有结果黑素瘤/cellranger/"
# path_cm_geo <- "E:/工作/论文/黑色素瘤/数据/收集的数据/GSE215120/"
# 
# # 样本列表
# am_samples <- c("WBZ1", "FGQI", "HYM1", "LY1", "SL1")
# cm_samples_geo <- c("GSM6622299_CM1", "GSM6622300_CM2", "GSM6622301_CM3")
# cm_zty1 <- "ZTY1"
# 
# # 读取 AM 样本
# seurat_am_list <- lapply(am_samples, function(name) {
#   path <- file.path(path_am, paste0(name, "-filtered_feature_bc_matrix.h5"))
#   counts <- Read10X_h5(path)
#   obj <- CreateSeuratObject(counts = counts, project = name)
#   obj$sample <- name
#   obj$group <- "AM"
#   return(obj)
# })
# names(seurat_am_list) <- am_samples
# 
# # 读取 CM GEO 样本
# seurat_cm_list_geo <- lapply(cm_samples_geo, function(name) {
#   path <- file.path(path_cm_geo, paste0(name, "_filtered_feature_bc_matrix.h5"))
#   counts <- Read10X_h5(path)
#   obj <- CreateSeuratObject(counts = counts, project = name)
#   obj$sample <- name
#   obj$group <- "CM"
#   return(obj)
# })
# names(seurat_cm_list_geo) <- cm_samples_geo
# 
# # 读取 ZTY1 样本
# path_zty1 <- file.path(path_cm_zty1, paste0(cm_zty1, "-filtered_feature_bc_matrix.h5"))
# counts_zty1 <- Read10X_h5(path_zty1)
# seurat_zty1 <- CreateSeuratObject(counts = counts_zty1, project = "ZTY1")
# seurat_zty1$sample <- "ZTY1"
# seurat_zty1$group <- "CM"
# 
# # 合并所有样本
# seurat_cm_list <- c(seurat_cm_list_geo, list(ZTY1 = seurat_zty1))
# all_samples <- c(seurat_am_list, seurat_cm_list)
# 
# # 预处理
# all_samples <- lapply(all_samples, function(obj) {
#   obj <- NormalizeData(obj)
#   obj <- FindVariableFeatures(obj, nfeatures = 2000)
#   return(obj)
# })
# 
# # 整合特征 + PCA
# features <- SelectIntegrationFeatures(all_samples, nfeatures = 2000)
# all_samples <- lapply(all_samples, function(x) {
#   x <- ScaleData(x, features = features, verbose = FALSE)
#   x <- RunPCA(x, features = features, verbose = FALSE)
#   return(x)
# })
# 
# # 整合
# anchors <- FindIntegrationAnchors(object.list = all_samples, anchor.features = features)
# seurat_integrated <- IntegrateData(anchorset = anchors)
# saveRDS(seurat_integrated, file = "E:/工作/论文/黑色素瘤/数据/单细胞数据整合结果/seurat_integrated.rds")



rm(list = ls())

library(Seurat)
library(dplyr)
library(harmony)
library(Matrix)


# 加载Seurat提供的细胞周期基因
data("cc.genes.updated.2019")
s.genes <- cc.genes.updated.2019$s.genes
g2m.genes <- cc.genes.updated.2019$g2m.genes

# 设置路径
path_am <- "E:/工作/论文/黑色素瘤/数据/现有结果黑素瘤/cellranger/"
path_cm_zty1 <- "E:/工作/论文/黑色素瘤/数据/现有结果黑素瘤/cellranger/"
path_cm_geo <- "E:/工作/论文/黑色素瘤/数据/收集的数据/GSE215120/"

am_samples <- c("WBZ1", "FGQI", "HYM1", "LY1", "SL1")
cm_samples_geo <- c("GSM6622299_CM1", "GSM6622300_CM2", "GSM6622301_CM3")
cm_zty1 <- "ZTY1"

# 函数：读取并过滤样本
read_and_filter <- function(name, path, group, from_geo = FALSE) {
  h5_file <- if (from_geo) {
    file.path(path, paste0(name, "_filtered_feature_bc_matrix.h5"))
  } else {
    file.path(path, paste0(name, "-filtered_feature_bc_matrix.h5"))
  }
  counts <- Read10X_h5(h5_file)
  obj <- CreateSeuratObject(counts = counts, project = name, min.cells = 3)
  obj <- obj[Matrix::rowSums(GetAssayData(obj, slot = "counts") > 0) > 3, ]
  obj$sample <- name
  obj$group <- group
  obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")
  obj <- subset(obj, subset = nFeature_RNA > 400 & nFeature_RNA < 7000 &
                  nCount_RNA < 50000 & percent.mt < 20)
  obj <- NormalizeData(obj)
  return(obj)
}

# 加载样本
seurat_am_list <- lapply(am_samples, function(name) {
  read_and_filter(name, path_am, "AM", from_geo = FALSE)
})
seurat_cm_list_geo <- lapply(cm_samples_geo, function(name) {
  read_and_filter(name, path_cm_geo, "CM", from_geo = TRUE)
})
seurat_zty1 <- read_and_filter(cm_zty1, path_cm_zty1, "CM", from_geo = FALSE)

# 合并所有样本
all_samples <- c(seurat_am_list, seurat_cm_list_geo, list(ZTY1 = seurat_zty1))
names(all_samples) <- c(am_samples, cm_samples_geo, cm_zty1) 
seurat_merged <- merge(all_samples[[1]], y = all_samples[-1])



layer_names <- names(seurat_merged[["RNA"]]@layers)
count_layers <- grep("^counts\\.", layer_names, value = TRUE)
data_layers <- grep("^data\\.", layer_names, value = TRUE)

all_genes <- Reduce(union, lapply(count_layers, function(layer) rownames(GetAssayData(seurat_merged[["RNA"]], layer = layer))))
counts_list <- lapply(count_layers, function(layer) {
  mat <- GetAssayData(seurat_merged[["RNA"]], layer = layer)
  missing <- setdiff(all_genes, rownames(mat))
  if (length(missing) > 0) {
    zero <- Matrix(0, nrow = length(missing), ncol = ncol(mat), sparse = TRUE)
    rownames(zero) <- missing; colnames(zero) <- colnames(mat)
    mat <- rbind(mat, zero)
  }
  mat[all_genes, , drop = FALSE]
})
merged_counts <- do.call(cbind, counts_list)

data_list <- lapply(data_layers, function(layer) {
  mat <- GetAssayData(seurat_merged[["RNA"]], layer = layer)
  missing <- setdiff(all_genes, rownames(mat))
  if (length(missing) > 0) {
    zero <- Matrix(0, nrow = length(missing), ncol = ncol(mat), sparse = TRUE)
    rownames(zero) <- missing; colnames(zero) <- colnames(mat)
    mat <- rbind(mat, zero)
  }
  mat[all_genes, , drop = FALSE]
})
merged_data <- do.call(cbind, data_list)

seurat_merged[["RNA"]] <- SetAssayData(seurat_merged[["RNA"]], layer = "counts", new.data = merged_counts)
seurat_merged[["RNA"]] <- SetAssayData(seurat_merged[["RNA"]], layer = "data", new.data = merged_data)




# 标准预处理 + 细胞周期回归 + 降维 + 整合 + 聚类
DefaultAssay(seurat_merged) <- "RNA"
seurat_merged <- NormalizeData(seurat_merged)
seurat_merged <- FindVariableFeatures(seurat_merged, nfeatures = 2000)

# 细胞周期打分与回归
seurat_merged <- CellCycleScoring(seurat_merged, s.features = s.genes, g2m.features = g2m.genes, set.ident = TRUE)
seurat_merged$CC.Difference <- seurat_merged$S.Score - seurat_merged$G2M.Score
seurat_merged <- ScaleData(seurat_merged, vars.to.regress = c("S.Score", "G2M.Score"))

# PCA + Harmony + 聚类 + UMAP
seurat_merged <- RunPCA(seurat_merged, npcs = 50)
seurat_merged <- RunHarmony(seurat_merged, group.by.vars = "sample")
seurat_merged <- FindNeighbors(seurat_merged, reduction = "harmony", dims = 1:50)
seurat_merged <- FindClusters(seurat_merged, resolution = 0.8, algorithm = 4)  # Leiden算法
seurat_merged <- RunUMAP(seurat_merged, reduction = "harmony", dims = 1:50)


DefaultLayer(seurat_merged[["RNA"]]) <- "data"

# 删除不需要的分层，只保留统一层
keep_layers <- intersect(names(seurat_merged[["RNA"]]@layers), c("counts", "data", "scale.data"))
seurat_merged[["RNA"]]@layers <- seurat_merged[["RNA"]]@layers[keep_layers]



# 保存结果
# saveRDS(seurat_merged, file = "E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/data/result/seurat_integrated.rds")





