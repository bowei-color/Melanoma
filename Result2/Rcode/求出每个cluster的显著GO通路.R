setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result2/code/")

rm(list = ls())

library(dplyr) 
library(clusterProfiler)
library(org.Hs.eg.db)  

markers <- read.csv("../data/source/melanocyte_subcluster_markers.csv")
cluster_ids <- unique(markers$cluster)
go_results_list <- list()
cluster_go_summary <- data.frame()

for (cl in cluster_ids) {
  genes <- markers %>%
    filter(cluster == cl) %>%
    arrange(desc(avg_log2FC)) %>%
    head(100) %>%
    pull(gene)
  
  go <- tryCatch({
    enrichGO(gene = genes,
             OrgDb = org.Hs.eg.db,
             keyType = "SYMBOL",
             ont = "BP",
             pAdjustMethod = "BH",
             pvalueCutoff = 0.05,
             qvalueCutoff = 0.2,
             readable = TRUE)
  }, error = function(e) NULL)
  
  go_results_list[[as.character(cl)]] <- go
  
  if (!is.null(go) && nrow(go) > 0) {
    cluster_go_summary <- rbind(cluster_go_summary, data.frame(
      cluster = cl,
      Description = go@result$Description[1],
      pvalue = go@result$p.adjust[1],
      stringsAsFactors = FALSE
    ))
  }
}



all_go_filtered <- data.frame()

for (cl in names(go_results_list)) {
  go <- go_results_list[[cl]]
  
  if (!is.null(go) && nrow(go@result) > 0) {
    filtered <- go@result %>%
      filter(p.adjust < 0.01) %>%
      mutate(cluster = cl)
    
    all_go_filtered <- rbind(all_go_filtered, filtered)
  }
}

write.csv(all_go_filtered, file = "../data/result/GO显著通路汇总_p小于0.01.csv", row.names = FALSE)
write.csv(cluster_go_summary, file = "../data/result/每个亚群最显著GO通路简表.csv", row.names = FALSE)

 