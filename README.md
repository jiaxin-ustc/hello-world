## 工作简介：
这段代码使用 keras(tensorflow) 框架，进行分类问题->回归问题的***迁移学习***性能探索。   
回归问题可以看作是分类问题更细粒度的特定任务；同时，分类相较于回归对标注的要求更低，因此可用于分类训练的数据集也更加多样、丰富。     
我们预期通过在分类任务 _(task=object+dataset)_ 上进行基础模型训练，随后进行回归任务训练的方式，实现性能上的迁移。   
这实际上可以是_Transfer+Fine-tuning_的一种实践，而这种行为的性能增益在早期的探索中得到了研究（ _Yosinski J et al, Advances in neural information processing systems, 2014_ ），并在后来成为一种常见且流行的一种范式。   

下面是思路流程图：  

![image](https://github.com/user-attachments/assets/e69db502-7143-403c-9c1e-0eacc3f65cfd)
    
这个工作在后来COVID-19来临时，被应用到“老药新用”的场景中：    
[Prediction of potential commercially available inhibitors against sars-cov-2 by multi-task deep learning model](https://pmc.ncbi.nlm.nih.gov/articles/PMC9405964/)
    
并为后续工作做了基础，文章详见：   
1. [
A multi-task deep model for protein-ligand interaction prediction](https://www.researchgate.net/profile/Peng-Yin-33/publication/336413320_A_Multi-Task_Deep_Model_for_Protein-Ligand_Interaction_Prediction/links/5da040e892851c6b4bcb7b6c/A-Multi-Task-Deep-Model-for-Protein-Ligand-Interaction-Prediction.pdf)  
2. [Multi-PLI: interpretable multi‐task deep learning model for unifying protein–ligand interaction datasets](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00510-6)

