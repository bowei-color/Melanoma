setwd("E:/工作/论文/黑色素瘤/code/MelanomaRcode/result6/code/")


rm(list = ls())

library(ggplot2)
library(reshape2)
library(dplyr)


signature_list <- list(
  Exhausted = c("PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4"),
  Cytotoxicity = c("GZMA", "GZMB", "PRF1", "GNLY", "IFNG", "NKG7", "CST7", "TNFSF10")
)


# 读取数据
data <- read.csv("E:/工作/论文/黑色素瘤/code/MelanomaAMCM/MelanomaAMCM/data/result6/result/gse190113_cm_cluster_data.csv")
samples <- data[, 1]
labels <- data[, 2]
expr <- data[, -c(1, 2)]
group <- ifelse(labels == 1, "1", "2")

# 初始化结果列表
score_list <- list()
p_table <- data.frame(Signature = character(), p_value = numeric())

# 逐个 signature 处理
for (sig in names(signature_list)) {
  genes <- signature_list[[sig]]
  genes <- intersect(genes, colnames(expr))
  score <- rowMeans(expr[, genes, drop = FALSE])
  df <- data.frame(Sample = samples, group = group, Signature = sig, Score = score)
  
  # 计算 p 值
  p_val <- wilcox.test(Score ~ group, data = df)$p.value
  df$p_value <- p_val  # 为当前 signature 的所有行添加相同 p 值
  score_list[[sig]] <- df
  
  p_table <- rbind(p_table, data.frame(Signature = sig, p_value = p_val))
}

# 合并数据并添加 adjusted p 值
plot_data <- do.call(rbind, score_list)
adj_p_table <- p_table %>% mutate(adj_p_value = p.adjust(p_value, method = "fdr"))

plot_data <- merge(plot_data, adj_p_table, by = "Signature")


# 保存结果
write.csv(plot_data, "../data/result/gse190113_cm_cluster_exhausted_cytotoxic_score.csv", row.names = FALSE)

plot_data$group <- ifelse(plot_data$group == "1", "cluster1", "cluster2")

# 绘图
p <- ggplot(plot_data, aes(x = Signature, y = Score, fill = group)) +
  geom_violin(scale = "width", trim = FALSE, linewidth = 1) +
  geom_boxplot(width = 0.2, outlier.shape = NA, position = position_dodge(width = 0.9), linewidth = 1.2, show.legend = FALSE) +
  scale_fill_manual(values = c("cluster1" = "#f7c376", "cluster2" = "#6bb3c8")) +
  theme_bw(base_size = 16) +
  labs(x = "", y = "") +
  theme(
    panel.grid = element_blank(),
    panel.border = element_blank(),
    axis.line.x = element_line(size = 1.2, color = "black"),  # 增加下边框
    axis.line.y = element_line(size = 1.2, color = "black"),  # 增加左边框
    axis.ticks = element_line(size = 1.2, color = "black"),   # 坐标轴刻度线加粗
    axis.ticks.length = unit(0.4, "cm"),
    axis.text.x = element_text(angle = 0, size = 30, hjust = 0.5),
    axis.text.y = element_text(size = 30),  
    legend.title = element_blank(), 
    legend.text = element_text(size = 30),
    legend.position = "top"
  )+ 
  geom_text(
    data = plot_data %>% group_by(Signature) %>% slice(1),
    aes(x = Signature, y = max(Score) + 6, label = paste0("adjusted p = ", signif(adj_p_value, 3))),
    inherit.aes = FALSE,
    size = 8
  )

ggsave("../figure/gse190113_cm_cluster_状态评分小提琴图.pdf", plot = p, width = 7, height = 7, dpi = 600)

