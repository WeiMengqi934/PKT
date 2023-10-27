from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np


class KTDataset(Dataset):
    def __init__(self, seqLen_allSample, questionID_allSample, skillID_allSample, label_allSample, n_question, n_skill, opt):
        assert len(seqLen_allSample) == len(questionID_allSample) == len(skillID_allSample) == len(label_allSample)
        self.seq_lengths_all = seqLen_allSample
        self.questionID_allSample = questionID_allSample
        self.skillID_allSample = skillID_allSample
        self.correctness_all = label_allSample
        self.n_question = n_question
        self.n_skill = n_skill
        self.eyeSkill = np.eye(n_skill + 1)
        self.opt = opt

    def __len__(self):
        return len(self.seq_lengths_all)

    def __getitem__(self, index: int):
        effLen_1Sample = self.seq_lengths_all[index] - 1
        questionID_1Sample = np.array(self.questionID_allSample[index], dtype=np.int32)
        skillID_1Sample = np.array(self.skillID_allSample[index], dtype=np.int32)
        label_1Sample = np.array(self.correctness_all[index], dtype=np.int32)

        currQuestionAddLabel_1Sample = label_1Sample[:-1] * self.n_question + questionID_1Sample[:-1]
        currQuestionID_1Sample = questionID_1Sample[:-1]
        currSkillAddLabel_1Sample = label_1Sample[:-1] * self.n_skill + skillID_1Sample[:-1]
        currSkillID_1Sample = skillID_1Sample[:-1]
        currSkill_oneHot_1Sample = self.eyeSkill[skillID_1Sample[:-1]]
        currLabel_1Sample = label_1Sample[:-1]

        nextQuestionID_1Sample = questionID_1Sample[1:]
        nextSkillID_1Sample = skillID_1Sample[1:]
        nextSkill_oneHot_1Sample = self.eyeSkill[skillID_1Sample[1:]]
        nextLabel_1Sample = label_1Sample[1:]

        return torch.LongTensor([effLen_1Sample]), \
               torch.LongTensor(currQuestionAddLabel_1Sample), torch.LongTensor(currQuestionID_1Sample), \
               torch.LongTensor(currSkillAddLabel_1Sample), torch.LongTensor(currSkillID_1Sample), torch.FloatTensor(currSkill_oneHot_1Sample), \
               torch.FloatTensor(currLabel_1Sample), \
               torch.LongTensor(nextQuestionID_1Sample), \
               torch.LongTensor(nextSkillID_1Sample), torch.FloatTensor(nextSkill_oneHot_1Sample), \
               torch.FloatTensor(nextLabel_1Sample)


class PadSequence(object):
    def __call__(self, batch: List[Tuple[torch.Tensor]]):
        batch = sorted(batch, key=lambda y: y[0].shape[0], reverse=True)

        effLen = torch.cat([x[0] for x in batch])

        currQuestionAddLabel = torch.nn.utils.rnn.pad_sequence([x[1] for x in batch], batch_first=True, padding_value=0)
        currQuestionID = torch.nn.utils.rnn.pad_sequence([x[2] for x in batch], batch_first=True, padding_value=0)

        currSkillAddLabel = torch.nn.utils.rnn.pad_sequence([x[3] for x in batch], batch_first=True, padding_value=0)
        currSkillID = torch.nn.utils.rnn.pad_sequence([x[4] for x in batch], batch_first=True, padding_value=0)
        currSkill_oneHot = torch.nn.utils.rnn.pad_sequence([x[5] for x in batch], batch_first=True)

        currLabel = torch.nn.utils.rnn.pad_sequence([x[6] for x in batch], batch_first=True, padding_value=0)
        nextQuestionID = torch.nn.utils.rnn.pad_sequence([x[7] for x in batch], batch_first=True, padding_value=0)

        nextSkillID = torch.nn.utils.rnn.pad_sequence([x[8] for x in batch], batch_first=True, padding_value=0)
        nextSkill_oneHot = torch.nn.utils.rnn.pad_sequence([x[9] for x in batch], batch_first=True)

        nextLabel = torch.nn.utils.rnn.pad_sequence([x[10] for x in batch], batch_first=True, padding_value=0)

        return effLen, \
               currQuestionAddLabel, currQuestionID, \
               currSkillAddLabel, currSkillID,currSkill_oneHot,\
               currLabel, \
               nextQuestionID, \
               nextSkillID, nextSkill_oneHot, \
               nextLabel
