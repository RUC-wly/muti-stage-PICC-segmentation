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
from skimage import transform as sktsf
import random

class TraindataSet(Dataset):
    def __init__(self,h_root,l_root, state ='Train', k=0):
        self.state = state
        path_list = os.listdir(os.path.join(h_root))
        path_num = len(path_list)
        fold_num = int(path_num // 5)
        h_fold_1 = [os.path.join(h_root, filename) for filename in path_list][0:fold_num-1]
        l_fold_1 = [os.path.join(l_root, filename) for filename in path_list][0:fold_num - 1]

        h_fold_2 = [os.path.join(h_root, filename) for filename in path_list][fold_num:fold_num*2-1]
        l_fold_2 = [os.path.join(l_root, filename) for filename in path_list][fold_num:fold_num * 2 - 1]

        h_fold_3 = [os.path.join(h_root, filename) for filename in path_list][fold_num*2:fold_num*3-1]
        l_fold_3 = [os.path.join(l_root, filename) for filename in path_list][fold_num * 2:fold_num * 3 - 1]

        h_fold_4 = [os.path.join(h_root, filename) for filename in path_list][fold_num*3:fold_num*4-1]
        l_fold_4 = [os.path.join(l_root, filename) for filename in path_list][fold_num * 3:fold_num * 4 - 1]

        h_fold_5 = [os.path.join(h_root, filename) for filename in path_list][fold_num * 4:path_num]
        l_fold_5 = [os.path.join(l_root, filename) for filename in path_list][fold_num*4:path_num]

        h_fold = [h_fold_1,h_fold_2, h_fold_3, h_fold_4, h_fold_5 ]
        l_fold = [l_fold_1, l_fold_2, l_fold_3, l_fold_4, l_fold_5]
        self.train_set_h=[]
        self.train_set_l = []
        self.test_set=[]

        for i in range(0, 5):
            if i != k:
                for ii in range(0,len(h_fold[i])):
                    self.train_set_h.append(h_fold[i][ii])
                    self.train_set_l.append(l_fold[i][ii])

    def __len__(self):
        return len(self.train_set_h)

    def __getitem__(self, index):
        img_high_path = self.train_set_h[index]
        img_low_path = self.train_set_l[index]

        img_high_rr = np.load(img_high_path)

        label_h = img_high_rr['picc']
        label_h = sktsf.resize(label_h, (1024, 1024), mode='reflect', anti_aliasing=False)
        ret, label_h = cv2.threshold(label_h, 0.5, 1, cv2.THRESH_BINARY)

        img_low_rr = np.load(img_low_path)
        image_l = img_low_rr['original']
        image_l = norm(image_l)
        image1 = sktsf.resize(image_l, (512, 512), mode='reflect', anti_aliasing=False)


        if self.state == 'train':
            number = random.randint(1, 10)
            if number % 2 == 0:
                # 高斯噪声
                mean = 0
                var = 0.005
                noise = np.random.normal(mean, var ** 0.5, image1.shape)
                out_image = image1 + noise

                return out_image, label_h, img_high_path

        return image1,  label_h, img_high_path



class Testdataset(Dataset):
    def __init__(self, h_root,l_root, state='Test', k=0):
        self.state = state
        path_list = os.listdir(os.path.join(h_root))
        path_num = len(path_list)
        fold_num = int(path_num // 5)

        h_fold_1 = [os.path.join(h_root, filename) for filename in path_list][0:fold_num - 1]
        l_fold_1 = [os.path.join(l_root, filename) for filename in path_list][0:fold_num - 1]

        h_fold_2 = [os.path.join(h_root, filename) for filename in path_list][fold_num:fold_num * 2 - 1]
        l_fold_2 = [os.path.join(l_root, filename) for filename in path_list][fold_num:fold_num * 2 - 1]

        h_fold_3 = [os.path.join(h_root, filename) for filename in path_list][fold_num * 2:fold_num * 3 - 1]
        l_fold_3 = [os.path.join(l_root, filename) for filename in path_list][fold_num * 2:fold_num * 3 - 1]

        h_fold_4 = [os.path.join(h_root, filename) for filename in path_list][fold_num * 3:fold_num * 4 - 1]
        l_fold_4 = [os.path.join(l_root, filename) for filename in path_list][fold_num * 3:fold_num * 4 - 1]

        h_fold_5 = [os.path.join(h_root, filename) for filename in path_list][fold_num * 4:path_num]
        l_fold_5 = [os.path.join(l_root, filename) for filename in path_list][fold_num * 4:path_num]

        h_fold = [h_fold_1, h_fold_2, h_fold_3, h_fold_4, h_fold_5]
        l_fold = [l_fold_1, l_fold_2, l_fold_3, l_fold_4, l_fold_5]

        self.test_set_h = []
        self.test_set_l = []

        for i in range(0, 5):
            if i == k:
                for ii in range(0, len(h_fold[i])):
                    self.test_set_h.append(h_fold[i][ii])
                    self.test_set_l.append(l_fold[i][ii])

    def __len__(self):
        return len(self.test_set_l)

    def __getitem__(self, index):
        img_high_path = self.test_set_h[index]
        img_low_path = self.test_set_l[index]

        img_high_rr = np.load(img_high_path)

        label_h = img_high_rr['picc']
        label_h = sktsf.resize(label_h, (1024, 1024), mode='reflect', anti_aliasing=False)
        ret, label_h = cv2.threshold(label_h, 0.5, 1, cv2.THRESH_BINARY)

        img_low_rr = np.load(img_low_path)
        image_l = img_low_rr['original']
        image_l = norm(image_l)

        image_l = CLAHE(image_l)
        image1 = sktsf.resize(image_l, (512, 512), mode='reflect', anti_aliasing=False)


        return image1, label_h, img_high_path


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data,  label_h, image_path) in enumerate(train_loader):
        # 这两行只适用 Net(1,1)
        data = torch.unsqueeze(data, dim=1)

        data, label_h = data.to(device), label_h.to(device)
        data = data.float()
        label_h = label_h.float()

        output = model(data)
        output = torch.squeeze(output)
        label_h = torch.squeeze(label_h)

        loss = criterion1(output, label_h)
        train1_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train1 Epoch: {} ,  Loss:{:.6f}'.format(epoch, sum(train1_loss[0:(len(train1_loss))])/len(train1_loss)))
    loss_out1 = sum(train1_loss[0:(len(train1_loss)-1)])/len(train1_loss)
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


def test(model,device,test_loader):
     model.eval()
     test_loss=0

     with torch.no_grad():
         for batch_idx, (data, label_h, img_path) in enumerate(test_dataloader):

             ###这两行只适用于Unet(1,1)
             data = torch.unsqueeze(data, dim=1)

             data, label_h = data.to(device), label_h.to(device)
             data = data.float()
             label_h = label_h.float()

             output= model(data)
             output = torch.squeeze(output)
             label_h = torch.squeeze(label_h)

             loss = criterion1(output, label_h)

             test_losses.append(loss)

             test_loss += loss

         outlabel=output.cpu().detach().numpy()

         test_loss /=len(test_loader.dataset)
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

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    from model.UNet_UP_Global import UNet
    model = UNet(1, 1)
    net_name_ = 'Global_Unet_UP_PPM_F'

    # from model.ResNet_UP_Global import Res_Net_
    # model = Res_Net_(1, 1)
    # net_name_ = 'Global_ResNet_UP_PPM_F'

    for i in range(0, 5):
        net_name = net_name_ + str(i+1)
        writer = SummaryWriter('runs/' + net_name_+str(i+1))
        k_ = i

        h_path= r'/data/wly/data/Code/pyramid_data/mid_res/train/'
        l_path = r'/data/wly/data/Code/pyramid_data/low_res/train/'
        LR = 0.001
        seg_mode = 'low_res'

        # 数据读取部分
        train_dataset = TraindataSet(h_path,l_path, k=k_)
        test_dataset = Testdataset(h_path,l_path, k=k_)

        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        # 输出保存路径
        image_save_path = os.path.join(r'/data/wly/data/Code/train', net_name)
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

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-06)
        # print(scheduler.optimizer)

        for epoch in range(train_epoch, 60):
            lr_print = 'lr:  ' + str(optimizer.state_dict()['param_groups'][0]['lr'])
            print(lr_print)

            train(model, device, train_dataloader, optimizer, epoch)
            test(model, device, test_dataloader)
        writer.close()

    # net_name = 'Global_ResNet_F1'
    # writer = SummaryWriter('runs/Global_Unet_PPM_F1')
    # k_ = 0
