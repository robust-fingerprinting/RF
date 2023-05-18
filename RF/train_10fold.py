# encoding: utf8

from datetime import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import pre_recall
from sklearn.model_selection import StratifiedShuffleSplit
from models.RF import getRF
import csv
import const_rf as const

EPOCH = 30
BATCH_SIZE = 200
LR = 0.0005
if_use_gpu = 1
num_classes = const.num_classes
num_folds = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(fpath):
    train = np.load(fpath, allow_pickle=True).item()
    train_X, train_y = train['dataset'], train['label']
    train_y = train_y[:, np.newaxis]
    print(train_X.shape, train_y.shape)
    return train_X, train_y


def adjust_learning_rate(optimizer, echo):
    lr = LR * (0.2 ** (echo / EPOCH))
    for para_group in optimizer.param_groups:
        para_group['lr'] = lr


def get_result(output, true_y):
    pred_y = torch.max(output, 1)[1].data.numpy().squeeze()
    accuracy = (pred_y == true_y.numpy()).sum().item() * 1.0 / float(true_y.size(0))
    return pred_y, accuracy


def val(cnn, test_x, test_y, result_file, test_file):
    cnn.eval()
    test_data = Data.TensorDataset(test_x, test_y)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
    for step, (tr_x, tr_y) in enumerate(test_loader):
        tr_x = tr_x.to(device)
        test_output = cnn(tr_x)
        if if_use_gpu:
            test_output = test_output.cpu()
        pred_y, accuracy = get_result(test_output, tr_y)
        resultfile = open(result_file, 'a+')
        for i in range(len(tr_y)):
            resultfile.write(str(tr_y[i].numpy()) + ',' + str(pred_y[i]) + '\n')
        resultfile.close()
        print(accuracy)


def test_train_data(cnn, train_loader, train_file):
    cnn.eval()
    train_result = open(train_file, 'a+', newline="")
    train_csv_wirter = csv.writer(train_result)
    acc = []
    for step, (tr_x, tr_y) in enumerate(train_loader):
        test_output = cnn(tr_x)
        if if_use_gpu:
            test_output = test_output.cpu()
            tr_y = tr_y.cpu()
        pred_y, accuracy = get_result(test_output, tr_y)
        acc.append(accuracy)
        print(accuracy)
    train_csv_wirter.writerow([sum(acc) / len(acc)])
    train_result.close()


def control(feature_file, result_file, test_file):
    x, y = load_data(feature_file)

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    fold = 1

    for train_index, test_index in sss.split(X=x, y=y):
        cnn = getRF(num_classes)
        if if_use_gpu:
            cnn = cnn.cuda()

        optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=0.001)
        loss_func = nn.CrossEntropyLoss()

        train_x = torch.unsqueeze(torch.from_numpy(x[train_index]), dim=1).type(torch.FloatTensor)
        train_x = train_x.view(train_x.size(0), 1, 2, -1)
        train_y = torch.squeeze(torch.from_numpy(y[train_index])).type(torch.LongTensor)
        test_x = torch.unsqueeze(torch.from_numpy(x[test_index]), dim=1).type(torch.FloatTensor)
        test_x = test_x.view(test_x.size(0), 1, 2, -1)
        test_y = torch.squeeze(torch.from_numpy(y[test_index])).type(torch.LongTensor)

        train_data = Data.TensorDataset(train_x, train_y)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

        cnn.train()

        for epoch in range(EPOCH):

            adjust_learning_rate(optimizer, epoch)
            for step, (tr_x, tr_y) in enumerate(train_loader):
                batch_x = tr_x.to(device)
                batch_y = tr_y.to(device)
                output = cnn(batch_x)
                _, accuracy = get_result(output.cpu(), tr_y.cpu())

                del batch_x
                loss = loss_func(output, batch_y)
                if step % 100 == 0:
                    print(epoch, step, accuracy, loss.item())

                del batch_y

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del output

        val(cnn, test_x, test_y, result_file.format(fold), test_file.format(fold))

        del train_x
        del train_y
        del train_data
        del train_loader
        del optimizer
        del test_x
        del test_y
        del cnn
        torch.cuda.empty_cache()
        print('*' * 5 + str(fold) + '*' * 5)
        fold += 1

        if fold > num_folds:
            break


def main():
    # TODO: change the data file path
    defense = 'Undefened'
    feature_file = 'dataset/' + defense + '.npy'
    result_file = 'result/' + defense + '-{}.csv'
    res_anaFile = result_file[:-4] + '_ana.csv'
    test_file = result_file[:-4] + '_test.txt'

    control(feature_file, result_file, test_file)

    for i in range(1, num_folds + 1):
        acc = pre_recall.pre_recCall(result_file.format(i), res_anaFile.format(i), num_classes)
        print('avg acc:' + str(acc))


if __name__ == '__main__':
    start = time.time()
    main()
    print(time.time() - start)
