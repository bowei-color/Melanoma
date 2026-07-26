setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result4/code")


rm(list = ls())

library(Seurat)
library(dplyr)
library(Matrix)
library(readr)

# Step 1: 读取 Seurat 对象
seurat_obj1 <- readRDS("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/data/result/seurat_annotated_obj.rds")
colnames(seurat_obj1@meta.data)
print(unique(seurat_obj1@meta.data$group))
expr_mat <- GetAssayData(seurat_obj1, assay = "RNA", layer = "data")  # log-normalized counts
meta_data <- seurat_obj1@meta.data

# Step 2: 读取 marker 表格
marker_file <- "E:/工作/论文/黑色素瘤/code/MelanomaAMCM/MelanomaAMCM/data/result1/source/T细胞细分亚群各标记基因占比.csv"
marker_data <- read.csv(marker_file, check.names = FALSE)

ordered_celltypes <- marker_data[[1]]
genes <- colnames(marker_data)[-1]

# Step 3: 定义计算函数
compute_expr_summary <- function(expr_matrix, meta_data, marker_data, ordered_celltypes, genes) {
  expr_ratio <- marker_data
  expr_avg <- marker_data
  
  for (i in seq_along(ordered_celltypes)) {
    celltype <- ordered_celltypes[i]
    cell_names <- rownames(meta_data)[meta_data$celltype == celltype]
    
    if (length(cell_names) == 0) {
      expr_ratio[i, -1] <- NA
      expr_avg[i, -1] <- NA
      next
    }
    
    expr_subset <- expr_matrix[genes, cell_names, drop = FALSE]
    expr_ratio[i, -1] <- Matrix::rowMeans(expr_subset > 0)
    expr_avg[i, -1] <- Matrix::rowMeans(expr_subset)
  }
  
  return(list(ratio = expr_ratio, avg = expr_avg))
}

# Step 4: 运行函数
result_all <- compute_expr_summary(expr_mat, meta_data, marker_data, ordered_celltypes, genes)

# Step 5: 保存结果
write.csv(result_all$ratio, "E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/data/result/各细胞类型中标记基因的表达占比.csv", row.names = FALSE)
write.csv(result_all$avg,   "E:/工作/论文/黑色素瘤/code/MelanomaRcode/result1/data/result/各细胞类型中标记基因的平均表达值.csv", row.names = FALSE)

