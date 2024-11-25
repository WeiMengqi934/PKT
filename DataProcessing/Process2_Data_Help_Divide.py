import numpy as np
import pandas as pd
import csv

# todo:————————划分数据集——————————————
# 每四行为一组
# 打开待划分的总数据集
with open('../1_data_AKT_Ques/data_4/EdNet/EdNet_drop1000.csv', 'r') as f_data:
    data_all = f_data.read().split('\n')

while data_all[0] == '':
    data_all= data_all[1:]
while data_all[-1] == '':
    data_all = data_all[:-1]

szAll_4 = len(data_all)

# begin--四个为一组，将数据进行分组
data = []
i = 0
while i < szAll_4:
    data_4 = []
    data_4.append(data_all[i])
    data_4.append(data_all[i+1])
    data_4.append(data_all[i+2])
    data_4.append(data_all[i+3])

    data.append(data_4)
    i += 4

# print(data[2])
# end----------------------分组结束---------------------


num_group = round(szAll_4/4)
index_list = [i for i in range(num_group)]
# 打乱索引
np.random.shuffle(index_list)


# 确定每个数据集的数量
num_train = round(num_group*0.6)
num_eval = round(num_group*0.2)
num_test = num_group - num_train - num_eval
print("num_group: ",num_group)
print("num_train: ",num_train)
print("num_eval: ",num_eval)
print("num_test: ",num_test)

# 开始分
# 给数据集命名，先是train，再是eval，再是test
with open('../1_data_AKT_Ques/data_4/EdNet/train5.csv', 'w') as f:
# with open('../1_data_AKT_Ques/assist12/train1.csv', 'w') as f:
    for i in range(num_train):
        f.write(data[index_list[i]][0])
        f.write('\n')
        f.write(data[index_list[i]][1])
        f.write('\n')
        f.write(data[index_list[i]][2])
        f.write('\n')
        f.write(data[index_list[i]][3])
        f.write('\n')

with open('../1_data_AKT_Ques/data_4/EdNet/eval5.csv', 'w') as f:
    j=0
    while j < num_eval :
        f.write(data[index_list[j + num_train]][0])
        f.write('\n')
        f.write(data[index_list[j + num_train]][1])
        f.write('\n')
        f.write(data[index_list[j + num_train]][2])
        f.write('\n')
        f.write(data[index_list[j + num_train]][3])
        f.write('\n')
        j += 1

with open('../1_data_AKT_Ques/data_4/EdNet/test5.csv', 'w') as f:
    k=0
    while k < num_test:
        f.write(data[index_list[k + num_train + num_eval]][0])
        f.write('\n')
        f.write(data[index_list[k + num_train + num_eval]][1])
        f.write('\n')
        f.write(data[index_list[k + num_train + num_eval]][2])
        f.write('\n')
        f.write(data[index_list[k + num_train + num_eval]][3])
        f.write('\n')
        k += 1


