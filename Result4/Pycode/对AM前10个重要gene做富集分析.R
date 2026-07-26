setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result4/code")


rm(list = ls())

library(clusterProfiler)
library(org.Hs.eg.db)
library(tidyverse)

# 读取基因数据
df <- read.csv("E:/工作/论文/黑色素瘤/code/MelanomaAMCM/MelanomaAMCM/data/result4/result/shapley解释得到的AM重要特征_averages.csv")
gene_symbols <- df$genesymbol %>% na.omit()

# 转换为 Entrez ID
gene_ids <- bitr(gene_symbols, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)

# 提示无法映射的基因
missing <- setdiff(gene_symbols, gene_ids$SYMBOL)
if (length(missing) > 0) warning("未能映射的基因：", paste(missing, collapse = ", "))

# GO BP 富集分析
ego <- enrichGO(gene = gene_ids$ENTREZID,
                OrgDb = org.Hs.eg.db,
                ont = "BP",
                pAdjustMethod = "BH",
                pvalueCutoff = 0.05,
                readable = TRUE)

# 柱状图（前10条）
barplot(ego, showCategory = 10, title = "GO BP Enrichment", font.size = 14)


dotplot(ego, showCategory = 10, title = "GO BP Enrichment", font.size = 14)

pdf("../figure/AM前10个基因重要基因GO富集气泡图.pdf",  width = 10, height = 10)
dotplot(ego, showCategory = 10, font.size  = 22) + 
  ggtitle("Top 10 important genes for AM")+
  theme(
    plot.title = element_text(size = 22),
    axis.text.y = element_text(size = 22, lineheight = 0.7), 
    axis.text.x = element_text(size = 22),
    axis.title = element_text(size = 22),
    legend.text = element_text(size = 22),
    legend.title = element_text(size = 22)
  )
dev.off() 


ego_df <- as.data.frame(ego)
write.csv(ego_df, "../data/result/AM重要基因_GO富集结果.csv", row.names = FALSE)
