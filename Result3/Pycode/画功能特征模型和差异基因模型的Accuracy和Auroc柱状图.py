# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 18:10:34 2025

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


test_data = pd.read_csv('../../data/result3/result/test_results.csv')
test_deg_data = pd.read_csv('../../data/result3/result/test_results_deg.csv')



methods = test_data['Model'].tolist()
x = np.arange(len(methods))  # 
width = 0.35  

colors_test = ['#d8d8d5'] * len(methods)         
colors_deg_test = ['#fad2a5'] * len(methods)     




# =============================================================================
# 画model和model_deg的Accuracy柱状图
# =============================================================================

plt.figure(figsize=(20, 20))
plt.bar(x - width/2, test_data['Accuracy'], width, label='Test(n=17)', color=colors_test, edgecolor='black', linewidth=6)
plt.bar(x + width/2, test_deg_data['Accuracy'], width, label='Test-DEG(n=66)', color=colors_deg_test, edgecolor='black', linewidth=6)
plt.ylabel('Accuracy', fontsize=80, labelpad=20)
plt.title('', fontsize=50)
plt.yticks(fontsize=70)
plt.tick_params(length=30, width=10, pad=10)
plt.ylim(0, 1)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(8)
ax.spines['bottom'].set_linewidth(8)
ax.set_xticks(x)
ax.set_xticklabels([''] * len(x))
for i, method in enumerate(methods):
    ax.text(x[i] + 0.4, -0.04, method, fontsize=70, ha='right', va='top', rotation=45)
# legend = plt.legend(fontsize=70, loc='upper right', bbox_to_anchor=(1.05, 1.2), ncol=2)
legend = plt.legend(fontsize=70, loc='upper right', bbox_to_anchor=(0.9, 1.25))
legend.get_frame().set_linewidth(0)  
legend.get_frame().set_facecolor('none') 
plt.savefig('../../figure/result3/model和model_deg在测试集上的Accuracy.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()



# =============================================================================
# 画model和model_deg的Accuracy柱状图
# =============================================================================

plt.figure(figsize=(20, 20))
plt.bar(x - width/2, test_data['AUC'], width, label='Test(n=17)', color=colors_test, edgecolor='black', linewidth=6)
plt.bar(x + width/2, test_deg_data['AUC'], width, label='Test-DEG(n=66)', color=colors_deg_test, edgecolor='black', linewidth=6)
plt.ylabel('Auroc', fontsize=80, labelpad=20)
plt.title('', fontsize=50)         
plt.yticks(fontsize=70)
plt.tick_params(length=30, width=10, pad=10)
plt.ylim(0, 1)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(8)
ax.spines['bottom'].set_linewidth(8)
ax.set_xticks(x)
ax.set_xticklabels([''] * len(x))
for i, method in enumerate(methods):
    ax.text(x[i] + 0.4, -0.04, method, fontsize=70, ha='right', va='top', rotation=45)
# legend = plt.legend(fontsize=70, loc='upper right', bbox_to_anchor=(1.05, 1.2), ncol=2)
legend = plt.legend(fontsize=70, loc='upper right', bbox_to_anchor=(0.9, 1.25))
legend.get_frame().set_linewidth(0)  
legend.get_frame().set_facecolor('none') 
plt.savefig('../../figure/result3/model和model_deg在测试集上的Auroc.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()