data_folder = 'datasets'

def get_csv_fname(trainEvalTest, dataset, datasetNum):
    if trainEvalTest == 'train':
        fname = 'train%s.csv'%datasetNum
    elif trainEvalTest == 'eval':
        fname = 'eval%s.csv'%datasetNum
    else:
        fname = 'test%s.csv'%datasetNum
    return '%s/%s/%s' % (data_folder, dataset, fname)


def get_num_skill(dataset):
    if dataset == 'assist0910':
        return 110
    elif dataset == 'assist17':
        return 102
    elif dataset == 'statics11':
        return 110
    elif dataset == 'EdNet':
        return 1849
    elif dataset == 'Eedi':
        return 700
    elif dataset == 'fsaif1tof3':
        return 1148
    else:
        raise NotImplementedError('Invalid Dataset')


def get_num_question(dataset):
    if dataset == 'assist0910':
        return 16891
    elif dataset == 'assist17':
        return 3162
    elif dataset == 'statics11':
        return 633
    elif dataset == 'EdNet':
        return 11848
    elif dataset == 'Eedi':
        return 950
    elif dataset == 'fsaif1tof3':
        return 24320
    else:
        raise NotImplementedError('Invalid Dataset')


def read_csv(fname, minimum_seq_len):
    with open(fname, 'r') as f:
        data = f.read()
    data = data.split('\n')
    while data[0] == '':
        data = data[1:]
    while data[-1] == '':
        data = data[:-1]
    effLen = []
    questionID = []
    skillID = []
    label = []
    i = 0
    while i < len(data):
        line = data[i]
        if i % 4 == 0:
            if int(line) >= minimum_seq_len:
                effLen.append(int(line))
            else:
                i += 4
                continue
        elif i % 4 == 1:
            line = line.split(',')
            questionID.append([int(e) for e in line if e != ''])
        elif i % 4 == 2:
            line = line.split(',')
            skillID.append([int(e) for e in line if e != ''])
        else:
            line = line.split(',')
            label.append([int(e) for e in line if e != ''])
            assert effLen[-1] == len(questionID[-1]) == len(skillID[-1]) == len(label[-1])
        i += 1
    return effLen, questionID, skillID, label
