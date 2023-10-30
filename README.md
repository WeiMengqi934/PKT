# PKT
Progressive knowledge tracing: Modeling learning process from abstract to concrete

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
