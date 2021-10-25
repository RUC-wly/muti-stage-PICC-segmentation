import os
from config.test_function import *
import numpy as np
import cv2

from myDataSetDiffSizeCrop import *
from skimage import transform as sktsf
from sklearn import metrics

def m_l_test(m_path, l_path, model, net_name):
    image_save_path = os.path.join(r'/data/wly/data/Code/test/', net_name)
    if os.path.exists(image_save_path):
        print('ImageSavePath : {}  已存在\n'.format(image_save_path))
    else:
        os.mkdir(image_save_path)
        print('ImageSavePath : {}  创建成功\n'.format(image_save_path))
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

    p_list =[]
    r_list =[]
    f_list = []
    a_list = []
    j_list = []
    fnb_list = []
    sensitivity_list = []
    iou_list = []
    Miou_list = []

    img_list = os.listdir(l_path)
    for i_l in img_list:
        l_img_path = os.path.join(l_path, i_l)
        m_img_path = os.path.join(m_path,i_l)
        img_o = np.load(l_img_path)['original']
        label_m = np.load(m_img_path)['picc']

        img = sktsf.resize(img_o, (512, 512), mode='reflect', anti_aliasing=False)
        label_m = sktsf.resize(label_m, (1024, 1024), mode='reflect', anti_aliasing=False)

        img = torch.tensor(img)
        img = torch.unsqueeze(img, dim=0)
        img = torch.unsqueeze(img, dim=0)

        img = img.to(device)
        img = img.float()
        seg_ = model(img)
        seg_ = torch.squeeze(seg_).cpu().detach().numpy()

        ret, seg = cv2.threshold(seg_, 0.5, 1, cv2.THRESH_BINARY)

        ret, label = cv2.threshold(label_m, 0.1, 1, cv2.THRESH_BINARY)

        TP = np.sum(np.logical_and(np.equal(label, 1), np.equal(seg, 1)))
        FP = np.sum(np.logical_and(np.equal(label, 0), np.equal(seg, 1)))
        FN = np.sum(np.logical_and(np.equal(label, 1), np.equal(seg, 0)))
        TN = np.sum(np.logical_and(np.equal(label, 0), np.equal(seg, 0)))

        fnb = FN / (FN + FP + TP)
        fnb_ = np.mean(fnb)
        fnb_list.append(fnb_)

        sensitivity = TP / (TP + FN)
        sensitivity_ = np.mean(sensitivity)
        sensitivity_list.append(sensitivity_)

        # specificity = TN /(TN+FP)
        # specificity_ = np.mean(specificity)
        # specificity_list.append(specificity_)

        iou = TP / (FP + TP + FN)
        iou_ = np.mean(iou)
        iou_list.append(iou_)

        Miou = (TP / (FP + TP + FN) + TN / (TN + FN + FP)) / 2
        Miou_ = np.mean(Miou)
        Miou_list.append(Miou_)

        p = metrics.precision_score(label, seg, average='micro')
        # print(p)

        recall = metrics.recall_score(label, seg, average='micro')
        # print(recall)

        f1 = metrics.f1_score(label, seg, average='micro')
        # print(f1)

        acc = metrics.accuracy_score(label, seg)
        # print(acc)

        jaa = metrics.jaccard_score(label, seg, average='micro' )
        # print(jaa)

        p_list.append(p)
        r_list.append(recall)
        f_list.append(f1)
        a_list.append(acc)
        j_list.append(jaa)

        label_name = image_save_path + '/' + i_l + '-label' + '.jpg'
        seg_name = image_save_path + '/' + i_l + '-seg_result' + '.jpg'
        data_name = image_save_path + '/' + i_l + '-data' + '.jpg'
        cv2.imwrite(label_name, norm(label) * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(seg_name, norm(seg_) * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(data_name, img_o * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    print("sensitivity:")
    print(np.mean(sensitivity_list))

    print("iou:")
    print(np.mean(iou_list))

    print("Miou:")
    print(np.mean(Miou_list))

    print("fnb:")
    print(np.mean(fnb_list))

    # for i in range(0,len(p_list)):
    #     print(p_list[i])
    #
    # print('recall')
    #
    # for i in range(0,len(p_list)):
    #     print(r_list[i])
    # print('f1')
    #
    # for i in range(0,len(p_list)):
    #     print(f_list[i])
    # print('acc')
    #
    # for i in range(0,len(p_list)):
    #     print(a_list[i])
    # print('jaa')
    #
    # for i in range(0,len(p_list)):
    #     print(j_list[i])



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # 超参部分

    test_lr_path = r'/data/wly/data/Code/pyramid_data/low_res/test/'
    test_mr_path = r'/data/wly/data/Code/pyramid_data/mid_res/test/'

    from model.UNet_UP_Global import UNet
    model = UNet(1, 1)
    net_name = "Global_Unet_UP_PPM_F5"

    # from model.ResNet_UP_Global import Res_Net_
    # model = Res_Net_(1, 1)
    # net_name = "Global_ResNet_UP_PPM_F1"

    m_l_test(test_mr_path, test_lr_path, model, net_name)