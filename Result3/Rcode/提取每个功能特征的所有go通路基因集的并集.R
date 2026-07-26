setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result3/code/")


rm(list = ls())

go_data <- read.csv("../../result2/data/result/GO显著通路汇总_p小于0.01.csv")

df <- read.csv("../../result2/data/source/代表性功能通路_用于ssGSEA.csv", stringsAsFactors = FALSE)

category_to_signature <- c(
  Immune_Response = "Signature1",
  Cell_Cycle = "Signature2",
  Translation_Ribosome = "Signature3",
  Stress_Response = "Signature4"
)

df$Signature <- category_to_signature[df$Functional_Category]

signature_map <- split(df$Description, df$Signature)

result_list <- list()

for (sig in names(signature_map)) {
  terms <- signature_map[[sig]]
  subset_rows <- go_data[go_data$Description %in% terms, ]
  for (i in seq_len(nrow(subset_rows))) {
    go_term <- subset_rows$Description[i]
    gene_vector <- unlist(strsplit(as.character(subset_rows$geneID[i]), "[/,]"))
    gene_vector <- trimws(gene_vector)
    result_list[[length(result_list) + 1]] <- data.frame(
      Signature = sig,
      GO_Term = go_term,
      Gene = gene_vector
    )
  }
}

final_df <- do.call(rbind, result_list)
write.csv(final_df, "../data/result/选定通路相关所有基因.csv", row.names = FALSE)



genes_sig1_sig1 <- unique(final_df$Gene[final_df$Signature %in% c("Signature1")])
genes_sig3_sig2 <- unique(final_df$Gene[final_df$Signature %in% c("Signature2")])
genes_sig1_sig3 <- unique(final_df$Gene[final_df$Signature %in% c("Signature3")])
genes_sig3_sig4 <- unique(final_df$Gene[final_df$Signature %in% c("Signature4")])

write.csv(data.frame(Gene = genes_sig1_sig2), "../data/result/Signature1相关通路所有基因.csv", row.names = FALSE)
write.csv(data.frame(Gene = genes_sig3_sig4), "../data/result/Signature2相关通路所有基因.csv", row.names = FALSE)
write.csv(data.frame(Gene = genes_sig1_sig2), "../data/result/Signature3相关通路所有基因.csv", row.names = FALSE)
write.csv(data.frame(Gene = genes_sig3_sig4), "../data/result/Signature4相关通路所有基因.csv", row.names = FALSE)
