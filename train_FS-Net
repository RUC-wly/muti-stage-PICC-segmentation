import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
from myDataSetDiffSizeCrop import norm,CLAHE
import cv2
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import re

class TraindataSet(Dataset):
    def __init__(self,root,state ='Train', k=0):
        self.state = state
        path_list = os.listdir(os.path.join(root))
        path_num = len(path_list)
        fold_num = int(path_num // 5)
        fold_1 = [os.path.join(root, filename) for filename in path_list][0:fold_num-1]
        fold_2 = [os.path.join(root, filename) for filename in path_list][fold_num:fold_num*2-1]
        fold_3 = [os.path.join(root, filename) for filename in path_list][fold_num*2:fold_num*3-1]
        fold_4 = [os.path.join(root, filename) for filename in path_list][fold_num*3:fold_num*4-1]
        fold_5 = [os.path.join(root, filename) for filename in path_list][fold_num*4:path_num]

        fold =[fold_1,fold_2, fold_3, fold_4, fold_5 ]
        self.train_set=[]
        self.test_set=[]

        for i in range(0, 5):
            if i != k:
                for ii in range(0,len(fold[i])):
                    self.train_set.append(fold[i][ii])

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, index):
        img_high_path = self.train_set[index]

        img_high_rr = np.load(img_high_path)
        image_h = img_high_rr['original']
        label_h = img_high_rr['label']

        seg_pre = image_h[1]
        choice = np.where(seg_pre > 0.1)
        xmin = choice[0].min()
        xmax = choice[0].max()
        ymin = choice[1].min()
        ymax = choice[1].max()

        if xmax > 512:
            x1 = xmax - 512
            x2 = xmax
        else:
            x1 = xmin
            x2 = xmin + 512

        if ymax > 512:
            y1 = ymax - 512
            y2 = ymax
        else:
            y1 = ymin
            y2 = ymin + 512

        image_crop_1 = image_h[0][x1:x2, y1:y2]
        image_crop_2 = image_h[1][x1:x2, y1:y2]
        label_crop = label_h[x1:x2, y1:y2]
        label_crop = norm(label_crop)
        ret, label_crop = cv2.threshold(label_crop, 0.5, 1, cv2.THRESH_BINARY)
        image_crop_1 = CLAHE(image_crop_1)

        image_crop = np.array([image_crop_1, image_crop_2])

        size = 11

        label_blur = cv2.GaussianBlur(label_crop, (size, size), 0)

        return image_crop, label_crop,label_blur, img_high_path


# path = r'/home/wly/data/Code/sigmoid_mid_data'
# data = TraindataSet(path, k=0)

class Testdataset(Dataset):
    def __init__(self, root, state='Test', k=0):
        self.state = state
        path_list = os.listdir(os.path.join(root))
        path_num = len(path_list)
        fold_num = int(path_num // 5)
        fold_1 = [os.path.join(root, filename) for filename in path_list][0:fold_num - 1]
        fold_2 = [os.path.join(root, filename) for filename in path_list][fold_num:fold_num * 2 - 1]
        fold_3 = [os.path.join(root, filename) for filename in path_list][fold_num * 2:fold_num * 3 - 1]
        fold_4 = [os.path.join(root, filename) for filename in path_list][fold_num * 3:fold_num * 4 - 1]
        fold_5 = [os.path.join(root, filename) for filename in path_list][fold_num * 4:path_num]

        fold = [fold_1, fold_2, fold_3, fold_4, fold_5]
        self.train_set = []
        self.test_set = []

        for i in range(0, 5):
            if i == k:
                for ii in range(0, len(fold[i])):
                    self.test_set.append(fold[i][ii])

    def __len__(self):
        return len(self.test_set)

    def __getitem__(self, index):
        img_high_path = self.test_set[index]

        img_high_rr = np.load(img_high_path)
        image_h = img_high_rr['original']
        label_h = img_high_rr['label']

        seg_pre = image_h[1]
        choice = np.where(seg_pre > 0.1)
        xmin = choice[0].min()
        xmax = choice[0].max()
        ymin = choice[1].min()
        ymax = choice[1].max()

        if xmax > 512:
            x1 = xmax - 512
            x2 = xmax
        else:
            x1 = xmin
            x2 = xmin + 512

        if ymax > 512:
            y1 = ymax - 512
            y2 = ymax
        else:
            y1 = ymin
            y2 = ymin + 512

        image_crop_1 = image_h[0][x1:x2, y1:y2]
        image_crop_2 = image_h[1][x1:x2, y1:y2]
        label_crop = label_h[x1:x2, y1:y2]
        label_crop = norm(label_crop)
        ret, label_crop = cv2.threshold(label_crop, 0.5, 1, cv2.THRESH_BINARY)
        image_crop_1 = CLAHE(image_crop_1)

        image_crop = np.array([image_crop_1, image_crop_2])

        return image_crop, label_crop, img_high_path


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, label_blur, image_path) in enumerate(train_loader):

        data, target, label_blur = data.to(device), target.to(device), label_blur.to(device)
        data = data.float()
        target = target.float()
        label_blur = label_blur.float()

        output, mask = model(data)
        output = torch.squeeze(output)
        target = torch.squeeze(target)
        mask = torch.squeeze(mask)
        label_blur = torch.squeeze(label_blur)

        loss1 = criterion1(output, target)
        loss2 = criterion2(mask, label_blur)
        loss = loss1 + 0.5 * loss2

        train1_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train1 Epoch: {} ,  Loss:{:.6f}'.format(epoch, sum(train1_loss[0:(len(train1_loss))]) / len(train1_loss)))
    loss_out1 = sum(train1_loss[0:(len(train1_loss) - 1)]) / len(train1_loss)
    train1_loss.append(loss_out1)
    writer.add_scalars(seg_mode, tag_scalar_dict={'train_loss1': loss_out1}, global_step=epoch)
    state = {
        'state': model.state_dict(),
        'epoch': epoch
    }

    if epoch % 5 == 0:
        path = os.path.join(model_save_path2, "{}_{}.t7".format(net_name, epoch))
        torch.save(state, path)
    path2 = os.path.join(model_save_path2, "{}_{}.t7".format(net_name, 200))
    torch.save(state, path2)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data, target, img_path) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)
            data = data.float()
            target = target.float()

            output, mask = model(data)
            output = torch.squeeze(output)
            target = torch.squeeze(target)

            loss = criterion1(output, target)

            test_losses.append(loss)
            test_loss += loss

        outlabel = output.cpu().detach().numpy()

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        writer.add_scalars(seg_mode, tag_scalar_dict={'test_loss': test_loss}, global_step=epoch)
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

        p, f = os.path.split(img_path[0])
        num_ = re.split(r'(\.)', f)
        outputname = image_save_path2 + '/' + str(epoch) + '-' + str(num_[0]) + '.jpg'
        cv2.imwrite(outputname, norm(outlabel) * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


if __name__ == '__main__':
    train1_loss = []
    test_losses = []

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    from new_Model.UNet_SW1 import UNet
    from new_Model.UNet_SW1_PAM import UNet
    from new_Model.UNet_SW1_PAM_D import UNet as Net

    model = Net(2, 1)
    # SW1_F SW1_PAM_F SW1_PAM_D_F
    net_name_ = "SW1(k11)_PAM_D_F"


    for i in range(0,5):
        net_name = net_name_ + str(i+1)
        writer = SummaryWriter('runs/'+net_name_+ str(i + 1))
        k_ = i

        # path = r'/data/wly/data/Code/Global_UNet_Up_PPM_dataset/'
        path = r'/data/wly/data/Code/Global_Unet_UP_PPM_F5_dataset/'
        LR = 0.001
        seg_mode = 'mid_res'

        # 数据读取部分
        train_dataset = TraindataSet(path, k=k_)
        test_dataset = Testdataset(path, k=k_)

        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        # 输出保存路径
        image_save_path = os.path.join(r'/data/wly/data/Code/train/', net_name)
        if os.path.exists(image_save_path):
            print('ImageSavePath : {}  已存在\n'.format(image_save_path))
        else:
            os.mkdir(image_save_path)
            print('ImageSavePath : {}  创建成功\n'.format(image_save_path))

        image_save_path2 = os.path.join(image_save_path, seg_mode)
        if os.path.exists(image_save_path2):
            print('ImageSavePath : {}  已存在\n'.format(image_save_path2))
        else:
            os.mkdir(image_save_path2)
            print('ImageSavePath : {}  创建成功\n'.format(image_save_path2))

        # 训练模型保存路径
        model_save_path = os.path.join(r'/data/wly/data/Code/fcn_test/checkpoint', net_name)
        if os.path.exists(model_save_path):
            print('ModelSavePath : {}  已存在\n'.format(model_save_path))
        else:
            os.mkdir(model_save_path)
            print('ModelSavePath : {}  创建成功\n'.format(model_save_path))

        model_save_path2 = os.path.join(model_save_path, seg_mode)
        if os.path.exists(model_save_path2):
            print('ModelSavePath : {}  已存在\n'.format(model_save_path2))
        else:
            os.mkdir(model_save_path2)
            print('ModelSavePath : {}  创建成功\n'.format(model_save_path2))

        # 读取已训练模型
        model_load_list = os.listdir(model_save_path2)
        if len(model_load_list):
            num_list = []
            max_num_list = []
            for i in range(len(model_load_list)):
                num_ = re.split(r'(\_|\.)', model_load_list[i])
                num = num_[-3]
                num_list.append(num)
            max_num_list = [int(i) for i in num_list]
            max_num_list_ = np.sort(max_num_list)
            maxnum = max_num_list_[-1]
            checkpoint_path = os.path.join(model_save_path2, "{}_{}.t7".format(net_name, maxnum))

            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state'])
            train_epoch = checkpoint['epoch']
            print("已加载{}_{}.t7".format(net_name, maxnum))
        else:
            train_epoch = 0
            print('==>start from now')

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model.to(device)
        criterion1 = nn.BCELoss()
        criterion2 = nn.L1Loss(reduction='mean')


        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-06)
        # print(scheduler.optimizer)

        for epoch in range(train_epoch, 60):
            lr_print = 'lr:  ' + str(optimizer.state_dict()['param_groups'][0]['lr'])
            print(lr_print)

            train(model, device, train_dataloader, optimizer, epoch)
            test(model, device, test_dataloader)
        writer.close()

