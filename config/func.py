import numpy as np
import cv2
import os
import re

def norm(data):

    max = np.max(data)
    min = np.min(data)
    normal_data = (data-min)/(max-min)

    return normal_data


def CLAHE(image):
    image = norm(image) * 255
    image = image.astype('uint8')

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    output = clahe.apply(image)
    output = norm(output)
    return output


def load_best_model(net_name, mode):
    chec_path= r'E:\py_code\PICC_related\fcn_test\checkpoint'
    path = os.path.join(chec_path, net_name)
    path = os.path.join(path, mode)
    model_load_list = os.listdir(path)
    if len(model_load_list):
        num_list = []
        for i in range(len(model_load_list)):
            num_ = re.split(r'(\_|\.)', model_load_list[i])
            num = num_[-3]
            num_list.append(num)
        max_num_list = [int(i) for i in num_list]
        max_num_list_ = np.sort(max_num_list)
        maxnum = max_num_list_[-1]
        checkpoint_path = os.path.join(path, "{}_{}.t7".format(net_name, maxnum))

    return checkpoint_path


def creat_test_save_path(net_name, mode):
    save_image_path = r'E:\py_code\outputset\UnetSeveralTest\Test'
    save_image_path = os.path.join(save_image_path, net_name)
    if os.path.exists(save_image_path):
        print('ImageSavePath : {}  已存在\n'.format(save_image_path))
    else:
        os.mkdir(save_image_path)
        print('ImageSavePath : {}  创建成功\n'.format(save_image_path))
    save_image_path = os.path.join(save_image_path, mode)
    if os.path.exists(save_image_path):
        print('ImageSavePath : {}  已存在\n'.format(save_image_path))
    else:
        os.mkdir(save_image_path)
        print('ImageSavePath : {}  创建成功\n'.format(save_image_path))

    return save_image_path


def threhold_aver(image):
    # img_array = np.array(image).astype(np.float32)  # 转化成数组
    # I = img_array
    # zmax = np.max(I)
    # zmin = np.min(I)
    # tk = (zmax + zmin) / 2  # 设置初始阈值
    # b = 1
    # m, n = I.shape
    # zo=0
    # zb=0
    # while b == 0:
    #     ifg = 0
    #     ibg = 0
    #     fnum = 0
    #     bnum = 0
    #
    #     for i in range(1, m):
    #         for j in range(1, n):
    #             tmp = I(i, j)
    #             if tmp >= tk:
    #                 ifg = ifg + 1
    #                 fnum = fnum + int(tmp)
    #             else:
    #                 ibg = ibg + 1
    #                 bnum = bnum + int(tmp)  # 背景像素的个数以及像素值的总和
    #     # 计算前景和背景的平均值
    #     zo = int(fnum / ifg)
    #     zb = int(bnum / ibg)
    #     if tk == int((zo + zb) / 2):
    #         b = 0
    #     else:
    #         tk = int((zo + zb) / 2)
    # return tk

    ret, thre_1 = cv2.threshold(image, 0.1, 1, cv2.THRESH_BINARY)
    ret, thre_2 = cv2.threshold(image, 0.2, 1, cv2.THRESH_BINARY)
    ret, thre_3 = cv2.threshold(image, 0.3, 1, cv2.THRESH_BINARY)
    ret, thre_4 = cv2.threshold(image, 0.4, 1, cv2.THRESH_BINARY)
    ret, thre_5 = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)
    ret, thre_6 = cv2.threshold(image, 0.6, 1, cv2.THRESH_BINARY)
    ret, thre_7 = cv2.threshold(image, 0.7, 1, cv2.THRESH_BINARY)
    ret, thre_8 = cv2.threshold(image, 0.8, 1, cv2.THRESH_BINARY)
    ret, thre_9 = cv2.threshold(image, 0.9, 1, cv2.THRESH_BINARY)
    output = (thre_1 + thre_2 + thre_3 + thre_4 + thre_5 + thre_6 + thre_7 + thre_8 + thre_9) /9
    return output

