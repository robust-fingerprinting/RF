# encoding: utf8

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import os
from models.RF import getRF
import const_rf as const

EPOCH = 30
BATCH_SIZE = 200
LR = 0.0005
if_use_gpu = 1
num_classes = const.num_classes

def load_data(fpath):
    train = np.load(fpath, allow_pickle=True).item()
    train_X, train_y = train['dataset'], train['label']

    return train_X, train_y


def adjust_learning_rate(optimizer, echo):
    lr = LR * (0.2 ** (echo / EPOCH))
    for para_group in optimizer.param_groups:
        para_group['lr'] = lr


def val(cnn, test_x, test_y, result_file, test_file):
    cnn.eval()
    test_result = open(test_file, 'w+')
    test_data = Data.TensorDataset(test_x, test_y)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
    for step, (tr_x, tr_y) in enumerate(test_loader):
        test_output = cnn(tr_x)
        if if_use_gpu:
            test_output = test_output.cpu()
        pred_y, accuracy = get_result(test_output, tr_y)
        resultfile = open(result_file, 'w+')
        for i in range(len(tr_y)):
            resultfile.write(str(tr_y[i].numpy()) + ',' + str(pred_y[i]) + '\n')
        resultfile.close()
        test_result.write(str(accuracy) + '\n')
        print(accuracy)
    test_result.close()


def get_result(output, true_y):
    pred_y = torch.max(output, 1)[1].data.numpy().squeeze()
    accuracy = (pred_y == true_y.numpy()).sum().item() * 1.0 / float(true_y.size(0))
    return pred_y, accuracy


def control(feature_file):
    x, y = load_data(feature_file)
    cnn = getRF(num_classes)

    if if_use_gpu:
        cnn = cnn.cuda()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=0.001)
    loss_func = nn.CrossEntropyLoss()
    train_x = torch.unsqueeze(torch.from_numpy(x), dim=1).type(torch.FloatTensor)
    train_x = train_x.view(train_x.size(0), 1, 2, -1)
    train_y = torch.from_numpy(y).type(torch.LongTensor)

    train_data = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    cnn.train()

    for epoch in range(EPOCH):

        adjust_learning_rate(optimizer, epoch)
        for step, (tr_x, tr_y) in enumerate(train_loader):
            batch_x = Variable(tr_x.cuda())
            batch_y = Variable(tr_y.cuda())
            output = cnn(batch_x)
            _, accuracy = get_result(output.cpu(), tr_y.cpu())

            del batch_x
            loss = loss_func(output, batch_y)
            del batch_y
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del output

            if step % 100 == 0:
                print(epoch, step, accuracy, loss.item())

    torch.save(cnn.state_dict(), os.path.join(const.model_path, method + '.pth'))


if __name__ == '__main__':
    # TODO: change the data file path
    defense = 'Undefended'
    feature_file = '../countermeasure/dataset/' + defense + '.npy'
    method = defense
    control(feature_file)
