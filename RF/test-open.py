import numpy as np
import torch
from models.RF import getRF
import torch.utils.data as Data
import const_rf as const
import csv
import pre_recall
from torch.nn import functional as F

def load_data(fpath):
    train = np.load(fpath, allow_pickle=True).item()
    train_X, train_y = train['dataset'], train['label']
    # train_y = train_y[:, np.newaxis]
    print(train_X.shape, train_y.shape)
    return train_X, train_y


def load_model(class_num, path, device):
    model = getRF(class_num)
    model.load_state_dict(torch.load(path + '.pth'))
    model = model.to(device)
    return model

if __name__ == '__main__':
    # TODO: change the test dataset path
    matrix_test_datast = ['dataset/.npy']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # TODO: change the trained model path
    model_list = ['model/']

    for i, path in enumerate(matrix_test_datast):
        print(path)
        defense_model = load_model(const.num_classes_ow, model_list[0], device).eval()
        features, test_y = load_data(path)
        print(features.shape)

        test_x = torch.unsqueeze(torch.from_numpy(features), dim=1).type(torch.FloatTensor)
        test_x = test_x.to(device)
        test_y = torch.squeeze(torch.from_numpy(test_y)).type(torch.LongTensor)
        test_data = Data.TensorDataset(test_x, test_y)
        train_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

        website_res = []
        for v, (x, y) in enumerate(train_loader):
            website_output = F.softmax(defense_model(x), dim=1).cpu().squeeze().detach().numpy()
            cur = [y.item()]
            cur.extend(website_output.tolist())
            website_res.append(cur)

        # You can find the test result in result/
        cur_website_path = 'result/{}_open.csv'.format(path[8:-4])
        with open(cur_website_path, 'w+', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for item in website_res:
                writer.writerow(item)
        pre_recall.score_func_precision_recall(cur_website_path[:-4] + '_open_ana.csv', website_res, const.num_classes_ow - 1)
