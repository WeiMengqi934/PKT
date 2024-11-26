# PKT
Paper title:《Progressive knowledge tracing: Modeling learning process from abstract to concrete》  

Paper download link: https://www.sciencedirect.com/science/article/pii/S0957417423027823?via%3Dihub   

## News！
**Notice that our work has made further progress.** 

The paper "Remembering is Not Applying: Interpretable Knowledge Tracing for Problem-solving Processes" has engaged in a comparison with our PKT model, and the related achievements have been published in the top conference 2024 ACM MM. **(ACM International Conference on Multimedia)**

link:https://openreview.net/forum?id=gDXAv76iP2

**The fact that our model is being used as a benchmark for comparison indicates that our work is meaningful and has been recognized!  We welcome everyone to follow up on this research.**

**有研究者注意到了我们的工作并进一步取得了可喜的成果。**

《Remembering is Not Applying: Interpretable Knowledge Tracing for Problem-solving Processes》该论文和我们的PKT模型进行了对比，相关成果已经发表在顶会2024 ACM MM上。能够作为被对比的模型，也说明我们的工作是有意义的且被认可的，这也给我们注入了信心。

非常欢迎大家跟进研究！

## Data Format
First line: The total number of exercises done by the student;  
Second line: The sequence of question IDs corresponding to the exercises;  
Third line: The sequence of skill IDs corresponding to the exercises;  
Fourth line: The correctness of each exercise answered by the student.

```
9								
749	260	509	327	207	835	579	160	528
144	219	245	204	181	70	70	70	181
0	1	1	1	0	0	1	1	0
```


## Data Preparation
You can download the original data from the following link and then refer to the methods described in the paper for data preprocessing:  

ASSIST09: https://github.com/arghosh/AKT/tree/master/data/assist2009_pid  

ASSIST17: https://github.com/arghosh/AKT/tree/master/data/assist2017_pid  

EdNet: https://github.com/riiid/ednet  

Eedi: https://eedi.com/projects/neurips-education-challenge  

Static11: https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507  

FSAI-F1toF3: https://github.com/ckyeungac/DeepIRT/tree/master/data/fsaif1tof3

## Cite
If you find this code useful in your research then please consider citing the following paper:  
```
@article{sun2023progressive,
  title={Progressive knowledge tracing: Modeling learning process from abstract to concrete},
  author={Sun, Jianwen and Wei, Mengqi and Feng, Jintian and Yu, Fenghua and Zou, Rui and Li, Qing},
  journal={Expert Systems with Applications},
  pages={122280},
  year={2023},
  publisher={Elsevier}
}
```





