import torch
from P6_utils import calculate_auc, calculate_acc, selecting_mask, calculate_lossGlobal, calculate_lossParams


def train(model, dataloaders, optimizer, opt):
    def run_epoch(train_eval_test):
        loss_epoch = 0
        predictRow_epoch = []
        nextLabelRow_epoch = []

        n_batch = None
        for n_batch, batch in enumerate(dataloaders[train_eval_test], start=1):
            effLens_batch, \
                currQuestionAddLabel_batch, currQuestionID_batch, \
                currSkillAddLabel_batch, currSkillID_batch, currSkill_oneHot_batch, \
                currLabel_batch, \
                nextQuestionID_batch, \
                nextSkillID_batch, nextSkill_oneHot_batch, \
                nextLabel_batch = batch

            effLens_batch, \
                currQuestionAddLabel_batch, \
                currSkillAddLabel_batch, \
                nextQuestionID_batch, \
                nextSkillID_batch, nextSkill_oneHot_batch, \
                nextLabel_batch = effLens_batch.to(opt.DEVICE), \
                currQuestionAddLabel_batch.to(opt.DEVICE), \
                currSkillAddLabel_batch.to(opt.DEVICE), \
                nextQuestionID_batch.to(opt.DEVICE), \
                nextSkillID_batch.to(opt.DEVICE), nextSkill_oneHot_batch.to(opt.DEVICE), \
                nextLabel_batch.to(opt.DEVICE)

            bsz, maxLen_batch = currSkillAddLabel_batch.size()
            maskEffLen_batch = selecting_mask(effLen=effLens_batch, maxLen=maxLen_batch, opt=opt)
            nextLabelRow_batch = nextLabel_batch.masked_select(maskEffLen_batch)
            nextLabelRow_epoch.append(nextLabelRow_batch.detach().cpu())

            if train_eval_test == 'train':
                model.train()
                optimizer.zero_grad()
            else:
                model.eval()

            predict_batch, predictAllSkill_batch, [L_skill, G, S] = model.forward(currSkillAddLabel_batch, currQuestionAddLabel_batch, nextSkill_oneHot_batch, nextSkillID_batch, nextQuestionID_batch)

            if train_eval_test == 'train':
                lossD_batch = calculate_lossGlobal(predict_batch, nextLabel_batch, maskEffLen_batch)
                lossParam_batch, lossList = calculate_lossParams([L_skill, G, S], nextSkill_oneHot_batch, effLens_batch, opt)
                (lossD_batch + lossParam_batch).backward()

                if n_batch == 1:
                    print('lossD_batch=%d' % lossD_batch)

                optimizer.step()
                loss_epoch += lossD_batch.item()

            predictRow_batch = predict_batch.masked_select(maskEffLen_batch)
            predictRow_epoch.append(predictRow_batch.detach().cpu())


        nextLabelRow_epoch = torch.cat(nextLabelRow_epoch)
        loss_epoch /= n_batch
        predictRow_epoch = torch.cat(predictRow_epoch)
        auc_epoch = calculate_auc(predictRow_epoch, nextLabelRow_epoch)
        acc_epoch = calculate_acc(predictRow_epoch, nextLabelRow_epoch)

        if train_eval_test == 'train':
            print('epoch %3d %5s \t\t|| acc=%.3f  auc=%.3f' % (epoch, train_eval_test, acc_epoch, auc_epoch))
        else:
            print('epoch %3d %5s \t\t|| acc=%.3f  auc=%.3f' % (epoch, train_eval_test, acc_epoch, auc_epoch))

        return auc_epoch, acc_epoch

    bestEpoch = 1
    bestAucEval = 0
    aucTestList = []
    accTestList = []

    for epoch in range(1, opt.n_epoch + 1):

        aucTrain, accTrain = run_epoch('train')
        aucEval, accEval = run_epoch('eval')
        aucTest, accTest = run_epoch('test')


        aucTestList.append(aucTest)
        accTestList.append(accTest)
        if aucEval > bestAucEval:
            bestAucEval = aucEval
            bestEpoch = epoch

        if epoch - bestEpoch >= 30:
            break

    bestAucTest = aucTestList[bestEpoch - 1]
    bestAccTest = accTestList[bestEpoch - 1]
    print('bestEpoch=%d \n bestAucTest=%.6f \n bestAccTest=%.6f' % (bestEpoch, bestAucTest, bestAccTest))
