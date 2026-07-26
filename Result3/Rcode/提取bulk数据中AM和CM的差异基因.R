setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result3/code/")

rm(list = ls())


library(limma)

data <- read.csv("E:/工作/论文/黑色素瘤/code/MelanomaAMCM/MelanomaAMCM/data/result3/source/GSE190113_data.csv")



expr_matrix <- t(data[, -1])
colnames(expr_matrix) <- paste0("Sample", 1:ncol(expr_matrix))
group <- factor(data$label)
design <- model.matrix(~group)
fit <- lmFit(expr_matrix, design)
fit <- eBayes(fit)
res <- topTable(fit, coef = 2, number = Inf)
res_sig <- res[order(res$adj.P.Val), ][1:78, ]





res_sig$gene <- rownames(res_sig)
write.csv(res_sig[, c("gene", setdiff(colnames(res_sig), "gene"))],
          "../data/result/GSE190113中AM和CM差异基因_top78.csv",
          row.names = FALSE)


