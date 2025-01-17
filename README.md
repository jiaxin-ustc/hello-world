## hello-world
test repository
第一次更新添加的内容


## 工作简介：
这段代码使用 keras(tensorflow) 框架，进行分类问题->回归问题的***迁移学习***性能探索。
回归问题可以看作是分类问题更细粒度的特定任务；同时，分类相较于回归对标注的要求更低，因此可用于分类训练的数据集也更加多样、丰富。
我们预期通过在分类任务_(task=object+dataset)_上进行基础模型训练，随后进行回归任务训练的方式，实现性能上的迁移。
这实际上可以是_Transfer+Fine-tuning_的一种实践，而这种行为的性能增益在早期的探索中得到了研究（_Yosinski J et al, Advances in neural information processing systems, 2014_），并在后来成为一种常见且流行的一种范式。
下面是思路流程图：
![image](https://github.com/user-attachments/assets/e69db502-7143-403c-9c1e-0eacc3f65cfd)

这个工作在后来COVID-19来临时，被应用到“老药新用”的场景中：
Hu F, Jiang J, Yin P. Prediction of potential commercially available inhibitors against sars-cov-2 by multi-task deep learning model[J]. Biomolecules, 2022, 12(8): 1156.

并为后续工作做了基础，文章引用：
Jiang J, Hu F, Zhu M, et al. A multi-task deep model for protein-ligand interaction prediction[C]//2019 International Conference on Intelligent Informatics and Biomedical Sciences (ICIIBMS). IEEE, 2019: 28-31.
Hu F, Jiang J, Wang D, et al. Multi-PLI: interpretable multi‐task deep learning model for unifying protein–ligand interaction datasets[J]. Journal of cheminformatics, 2021, 13: 1-14.
第二次更新添加的内容
