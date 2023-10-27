from P0_config import set_opt
from P1_dataUtil import get_csv_fname, read_csv, get_num_skill, get_num_question
from P2_dataset import KTDataset, PadSequence
from P4_model_inSkillQues import DKTInSkillQues
from P5_trainUtil import train
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam


def trainComplementaryModel_main():
    opt.n_skill = get_num_skill(opt.dataset)
    opt.n_skillPad = opt.n_skill + 1
    opt.n_question = get_num_question(opt.dataset)
    opt.n_questionPad = opt.n_question + 1

    fnames = {'train': get_csv_fname('train', opt.dataset, opt.datasetNum), 'eval': get_csv_fname('eval', opt.dataset, opt.datasetNum), 'test': get_csv_fname('test', opt.dataset, opt.datasetNum)}
    datasets = {'train': read_csv(fnames['train'], opt.min_seq_len), 'eval': read_csv(fnames['eval'], opt.min_seq_len), 'test': read_csv(fnames['test'], opt.min_seq_len)}

    datasets = {'train': KTDataset(seqLen_allSample=datasets['train'][0],
                                   questionID_allSample=datasets['train'][1],
                                   skillID_allSample=datasets['train'][2],
                                   label_allSample=datasets['train'][3],
                                   n_question=opt.n_question, n_skill=opt.n_skill, opt=opt),
                'eval': KTDataset(datasets['eval'][0],
                                   datasets['eval'][1],
                                   datasets['eval'][2],
                                   datasets['eval'][3],
                                   opt.n_question, opt.n_skill, opt=opt),
                'test': KTDataset(datasets['test'][0],
                                  datasets['test'][1],
                                  datasets['test'][2],
                                  datasets['test'][3],
                                  opt.n_question, opt.n_skill, opt=opt)}

    dataloaders = {'train': DataLoader(dataset=datasets['train'],
                                       batch_size=opt.bsz,
                                       drop_last=False,
                                       collate_fn=PadSequence(),
                                       shuffle=opt.shuffleDataloader),
                   'eval': DataLoader(dataset=datasets['eval'],
                                       batch_size=opt.bsz,
                                       drop_last=False,
                                       collate_fn=PadSequence(),
                                       shuffle=opt.shuffleDataloader),
                   'test': DataLoader(dataset=datasets['test'],
                                      batch_size=opt.bsz,
                                      drop_last=False,
                                      collate_fn=PadSequence(),
                                      shuffle=opt.shuffleDataloader)}

    model=DKTInSkillQues(nSkill=opt.n_skill, nQues=opt.n_question, szRnnIn=opt.sz_rnn_in, szRnnOut=opt.sz_rnn_out, nRnnLayer=opt.n_rnn_layer, szOut=opt.n_skillPad, dropout=opt.rnn_dropout, opt=opt).to(opt.DEVICE)
    optimizer=Adam(model.parameters(), lr=opt.lr)
    train(model, dataloaders, optimizer, opt)


if __name__ == '__main__':
    opt = set_opt()
    torch.manual_seed(opt.seed)
    trainComplementaryModel_main()
