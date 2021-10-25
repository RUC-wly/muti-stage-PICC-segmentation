import os
import numpy as np
from skimage import transform as sktsf
import torch
import re
from torch.utils.data import DataLoader
from myDataSetDiffSizeCrop import Mid_res_Dataset_test,low_res_DataSet_test


def imglist_from_path(path):

    img_path_list = []
    path_list = os.listdir(path)
    for path_l in path_list:
        img_path_list.append(os.path.join(path, path_l))
    return img_path_list


def img_from_list(list_item):
    image_r = np.load(list_item)
    image = image_r['original']
    label = image_r['picc']
    return image, label


def m_l_model_test(model, net_name, m_path, l_path):
    check_path = r'/data/wly/data/Code/fcn_test/checkpoint/'
    seg_mode = 'low_res'
    model_check_path = os.path.join(check_path, net_name)
    model_check_path_ = os.path.join(model_check_path, seg_mode)
    model_load_list = os.listdir(model_check_path_)
    if len(model_load_list):
        num_list = []
        for i in range(len(model_load_list)):
            num_ = re.split(r'(\_|\.)', model_load_list[i])
            num = num_[-3]
            num_list.append(num)
        max_num_list = [int(i) for i in num_list]
        max_num_list_ = np.sort(max_num_list)
        maxnum = max_num_list_[-1]
        checkpoint_path = os.path.join(model_check_path_, "{}_{}.t7".format(net_name, maxnum))
    else:
        print('model is not exist!!!')

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state'])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model.to(device)
    test_dataset = Mid_res_Dataset_test(path_high=m_path, path_low=l_path, split_num=0, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    seg_pre = []
    label_pre = []
    data_ = []
    i = 0
    with torch.no_grad():
        for batch_idx, (data, target, image_h, label_h, img_path) in enumerate(test_dataloader):
            data = torch.unsqueeze(data, dim=1)

            # if i == 31:
            #     print(1)
            # i = i + 1
            data, label_h = data.to(device), label_h.to(device)
            data = data.float()
            label_h = label_h.float()

            output = model(data)
            output = torch.squeeze(output)
            label_h = torch.squeeze(label_h)

            image_h = torch.squeeze(image_h)

            outlabel = output.cpu().detach().numpy()
            label_h = label_h.cpu().detach().numpy()
            image_h = image_h.cpu().detach().numpy()
            seg_pre.append(outlabel)
            label_pre.append(label_h)
            data_.append(image_h)
    return seg_pre, label_pre, data_


def l_model_test(model, net_name, l_path):
    check_path = r'/data/wly/data/Code/fcn_test/checkpoint/'
    seg_mode = 'low_res'
    model_check_path = os.path.join(check_path, net_name)
    model_check_path_ = os.path.join(model_check_path, seg_mode)
    model_load_list = os.listdir(model_check_path_)
    if len(model_load_list):
        num_list = []
        for i in range(len(model_load_list)):
            num_ = re.split(r'(\_|\.)', model_load_list[i])
            num = num_[-3]
            num_list.append(num)
        max_num_list = [int(i) for i in num_list]
        max_num_list_ = np.sort(max_num_list)
        maxnum = max_num_list_[-1]
        checkpoint_path = os.path.join(model_check_path_, "{}_{}.t7".format(net_name, maxnum))
    else:
        print('model is not exist!!!')

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state'])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    test_dataset = low_res_DataSet_test(l_path, split_num=0, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle= False, num_workers=0)

    seg_pre = []
    label_pre = []
    data_ = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_dataloader):
            data = torch.unsqueeze(data, dim=1)
            data, target = data.to(device), target.to(device)
            data = data.float()
            target = target.float()

            output = model(data)
            output = torch.squeeze(output)
            target = torch.squeeze(target)
            data = torch.squeeze(data)
            data = data.cpu().detach().numpy()
            outlabel = output.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            seg_pre.append(outlabel)
            label_pre.append(target)
            data_.append(data)


    return seg_pre, label_pre, data_


def DICE(predict,label):

    input_flatten = predict.flatten()
    target_flatten = label.flatten()
    # 计算交集中的数量
    overlap = np.sum(input_flatten * target_flatten)
    # 返回值，让值在0和1之间波动
    return np.clip(((2. * overlap) / (np.sum(target_flatten) + np.sum(input_flatten) )), 1e-4, 0.9999)


def local_test(data_crop, seg_crop, model, net_name):
    data = np.array([data_crop, seg_crop])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data = torch.tensor(data)
    data = data.float()

    data = data.to(device)
    data = torch.unsqueeze(data, dim=0)

    check_path = r'/home/wly/wly/data/Code/fcn_test/checkpoint/'
    seg_mode = 'mid_res'

    model_check_path = os.path.join(check_path, net_name)
    model_check_path_ = os.path.join(model_check_path, seg_mode)
    model_load_list = os.listdir(model_check_path_)
    if len(model_load_list):
        num_list = []
        for i in range(len(model_load_list)):
            num_ = re.split(r'(\_|\.)', model_load_list[i])
            num = num_[-3]
            num_list.append(num)
        max_num_list = [int(i) for i in num_list]
        max_num_list_ = np.sort(max_num_list)
        maxnum = max_num_list_[-1]
        checkpoint_path = os.path.join(model_check_path_, "{}_{}.t7".format(net_name, maxnum))
    else:
        print('model is not exist!!!')

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state'])
    model.to(device)
    seg2,_ = model(data)
    # seg2 = model(data)

    seg2 = torch.squeeze(seg2).cpu().detach().numpy()

    return seg2

def local_test2(data_crop, seg_crop, model, net_name):
    data = np.array([data_crop, seg_crop])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data = torch.tensor(data)
    data = data.float()

    data = data.to(device)
    data = torch.unsqueeze(data, dim=0)

    check_path = r'/home/wly/wly/data/Code/fcn_test/checkpoint/'
    seg_mode = 'mid_res'

    model_check_path = os.path.join(check_path, net_name)
    model_check_path_ = os.path.join(model_check_path, seg_mode)
    model_load_list = os.listdir(model_check_path_)
    if len(model_load_list):
        num_list = []
        for i in range(len(model_load_list)):
            num_ = re.split(r'(\_|\.)', model_load_list[i])
            num = num_[-3]
            num_list.append(num)
        max_num_list = [int(i) for i in num_list]
        max_num_list_ = np.sort(max_num_list)
        maxnum = max_num_list_[-1]
        checkpoint_path = os.path.join(model_check_path_, "{}_{}.t7".format(net_name, maxnum))
    else:
        print('model is not exist!!!')

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state'])
    model.to(device)
    # seg2,_ = model(data)
    seg2 = model(data)

    seg2 = torch.squeeze(seg2).cpu().detach().numpy()

    return seg2

def findpoint(image):
    choice = np.where(image != 0)
    list = np.where(choice[0] == choice[0].max())
    maxy = choice[0].max()
    maxx = choice[1][list[len(list) - 1][-1]]
    cxmax = choice[1].max()
    cxmin = choice[1].min()
    cx = (cxmax + cxmin) / 2

    while maxx < cx:
        l = choice[0]
        l[l == maxy] = 0
        maxy = l.max()
        list = np.where(choice[0] == maxy)
        maxx = choice[1][list[len(list) - 1][-1]]
        cxmax = choice[1].max()
        cxmin = choice[1].min()
        cx = (cxmax + cxmin) / 2

    return maxy, maxx


import skimage.transform as st
import cv2
import cv2
import numpy as np

def line_patch(img):
    # kernel = np.ones((5, 5), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # 行扫描，间隔k时，进行填充，填充值为1
    def edge_connection(img, size, k):
        for i in range(size):
            Yi = np.where(img[i, :] > 0)
            if len(Yi[0]) >= 2:  # 可调整
                for j in range(0, len(Yi[0]) - 1):
                    if Yi[0][j + 1] - Yi[0][j] <= k:
                        img[i, Yi[0][j]:Yi[0][j + 1]] = 1
        return img

    g = edge_connection(img, img.shape[0], k=40)
    g = cv2.rotate(g, 0)
    g = edge_connection(g, img.shape[1], k=40)
    g = cv2.rotate(g, 2)

    # kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # cv2.imshow("img", img)
    # cv2.imshow("g", opening)
    # cv2.waitKey(0)
    # print(1)

    return g


from skimage import draw


def tip_location(mask, image):
    image = image * 255
    image = image.astype('uint8')

    y, x = findpoint(mask)
    cv2.circle(image, (x, y), 10, (0, 255, 255))
    i = cv2.add(image, mask.astype('uint8') * 255)

    return i, x, y




























