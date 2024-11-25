# 去除超过1000的，去除小于10的四行数据

import numpy as np

with open('../EdNet_all_que_skill.txt', 'r') as f_data:
    data_all = f_data.read().split('\n')

while data_all[0] == '':
    data_all= data_all[1:]
while data_all[-1] == '':
    data_all = data_all[:-1]

szAll_4 = len(data_all)

# print(data_all[0])


data = []
i = 0
while i < szAll_4:
    data_4 = []
    if int(data_all[i])<9 or int(data_all[i])>1000:
        i += 4
    else:
        data_4.append(data_all[i])
        data_4.append(data_all[i+1])
        data_4.append(data_all[i+2])
        data_4.append(data_all[i+3])

        data.append(data_4)
        i += 4

num_group = len(data)
# num_group = round(szAll_4/4)
index_list = [i for i in range(num_group)]

with open('../EdNet_drop1000.csv', 'w') as f:
    for i in range(num_group):
        f.write(data[index_list[i]][0])
        f.write('\n')
        f.write(data[index_list[i]][1])
        f.write('\n')
        f.write(data[index_list[i]][2])
        f.write('\n')
        f.write(data[index_list[i]][3])
        f.write('\n')




