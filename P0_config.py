from argparse import ArgumentParser
import torch


def set_opt():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='assist0910',
                        help='choose from assist0910, assist17, statics11, EdNet, Eedi, fsaif1tof3')
    parser.add_argument('--datasetNum', type=str, default='1')
    parser.add_argument('--shuffleDataloader', type=bool, default=True)
    parser.add_argument('--min_seq_len', type=int, default=2,
                        help='minimum threshold of number of time steps to discard student problem-solving records.')
    parser.add_argument('--n_epoch', type=int, default=600, help='number of epochs')
    parser.add_argument('--bsz', type=int, default=10)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use. default to -1: not using any')
    sz_rnn = 128
    parser.add_argument('--sz_rnn_in', type=int, default=sz_rnn)
    parser.add_argument('--sz_rnn_out', type=int, default=sz_rnn)
    parser.add_argument('--n_rnn_layer', type=int, default=1, help='number of rnn layers')
    parser.add_argument('--rnn_dropout', type=float, default=0.0)
    parser.add_argument('--L_forwardPunishThreshold', type=float, default=0.7)
    parser.add_argument('--L_backwardPunishThreshold', type=float, default=0.3)
    parser.add_argument('--LForwardPunish', type=int, default=1)
    parser.add_argument('--LBackPunish', type=int, default=1)
    parser.add_argument('--G_punishThresholdCoef', type=float, default=0.4)
    parser.add_argument('--S_punishThresholdCoef', type=float, default=0.4)
    opt = parser.parse_args()

    if opt.gpu == -1 or torch.cuda.is_available() == False:
        opt.DEVICE = 'cpu'
        opt.gpu = -1
    else:
        opt.DEVICE = opt.gpu
    print(opt)
    return opt
