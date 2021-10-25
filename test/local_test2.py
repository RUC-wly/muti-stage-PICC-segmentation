
from myDataSetDiffSizeCrop import *

from config.test_function import *



# global_up model
# return list:seg_pre, label 1024
def m_l_model_test(h_pth, l_path, model, net_name):
    check_path = r'/home/wly/wly/data/Code/fcn_test/checkpoint/'
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
    seg_pre = []
    label_pre = []

    test_dataset = Mid_res_Dataset_test(path_high=h_pth, path_low=l_path, split_num=0, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch_idx, (data, target, image_h, label_h, img_path) in enumerate(test_dataloader):
        data = torch.unsqueeze(data, dim=1)

        data, label_h = data.to(device), label_h.to(device)
        data = data.float()
        label_h = label_h.float()

        output = model(data)
        output = torch.squeeze(output)
        label_h = torch.squeeze(label_h)

        out = output.cpu().detach().numpy()
        label = label_h.cpu().detach().numpy()

        seg_pre.append(out)
        label_pre.append(label)

    return seg_pre, label_pre


# global model
# return list:seg_pre, label 1024
def l_model_test(l_path, m_path, model, net_name):
    seg_pre = []
    label_list = []
    check_path = r'/home/wly/wly/data/Code/fcn_test/checkpoint/'
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

    img_list = os.listdir(l_path)
    for i_l in img_list:
        l_img_path = os.path.join(l_path, i_l)
        m_img_path = os.path.join(m_path, i_l)
        img_o = np.load(l_img_path)['original']
        label_m = np.load(m_img_path)['picc']

        img = sktsf.resize(img_o, (512, 512), mode='reflect', anti_aliasing=False)
        label_m = sktsf.resize(label_m, (1024, 1024), mode='reflect', anti_aliasing=False)

        # debug watch the shape of img
        img = torch.tensor(img)
        img = torch.unsqueeze(img, dim=0)
        img = torch.unsqueeze(img, dim=0)

        img = img.to(device)
        img = img.float()
        seg_ = model(img)
        seg_ = torch.squeeze(seg_).cpu().detach().numpy()

        seg_end = sktsf.resize(seg_, (1024, 1024), mode='constant', anti_aliasing=False)
        ret, seg = cv2.threshold(seg_end, 0.1, 1, cv2.THRESH_BINARY)

        ret, label = cv2.threshold(label_m, 0.5, 1, cv2.THRESH_BINARY)
        seg_pre.append(seg)
        label_list.append(label)

    return seg_pre, label_list


def mid_res_local_test(m_path, l_path, model1, model2, net_name1, net_name2):
    image_save_path = os.path.join(r'/home/wly/wly/data/Code/ALL_test', net_name2)
    if os.path.exists(image_save_path):
        print('ImageSavePath : {}  已存在\n'.format(image_save_path))
    else:
        os.mkdir(image_save_path)
        print('ImageSavePath : {}  创建成功\n'.format(image_save_path))

    # Coarse-network with up_sample
    seg_pre, _ = m_l_model_test(l_path, m_path, model1, net_name1)

    # Course-network without up_sample
    # seg_pre, _ = l_model_test( l_path,m_path, model1, net_name1)
    data_ = []
    label_pre = []

    for p in os.listdir(m_path):
        mr_image_path = os.path.join(m_path, p)
        mr_image = np.load(mr_image_path)
        mr_data = mr_image['original']
        mr_data = sktsf.resize(mr_data, (1024, 1024), mode='reflect', anti_aliasing=False)
        mr_label = mr_image['picc']
        mr_label = sktsf.resize(mr_label, (1024, 1024), mode='reflect', anti_aliasing=False)
        data_.append(mr_data)
        label_pre.append(mr_label)

    num = len(seg_pre)

    for i in range(0, num):
        seg_pre_ = seg_pre[i]
        seg_pre_ = sktsf.resize(seg_pre_, (1024, 1024), mode='reflect', anti_aliasing=False)
        data = data_[i]
        data = CLAHE(data)

        choice = np.where(label_pre[i] > 0.1)
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

        data_crop = data[x1:x2, y1:y2]
        seg_crop = seg_pre_[x1:x2, y1:y2]
        local_seg = local_test(data_crop, seg_crop, model2, net_name2)
        seg1_with_seg2 = np.array(seg_pre_, copy=True)
        seg1_with_seg2[x1:x2, y1:y2] = local_seg[:, :]

        seg = sktsf.resize(seg1_with_seg2, (1024, 1024), mode='reflect', anti_aliasing=False)
        label = sktsf.resize(label_pre[i], (1024, 1024), mode='reflect', anti_aliasing=False)
        ret, new_label = cv2.threshold(label, 0.1, 1, cv2.THRESH_BINARY)

        TP_list = np.zeros((1, 100))
        FP_list = np.zeros((1, 100))
        FN_list = np.zeros((1, 100))
        TN_list = np.zeros((1, 100))

        for t in range(0, 100):
            t1 = t * 0.01
            ret, seg_t = cv2.threshold(seg, t1, 1, cv2.THRESH_BINARY)

            TN_t = np.sum(np.logical_and(np.equal(new_label, 0), np.equal(seg_t, 0)))
            TP_t = np.sum(np.logical_and(np.equal(new_label, 1), np.equal(seg_t, 1)))
            FP_t = np.sum(np.logical_and(np.equal(new_label, 0), np.equal(seg_t, 1)))
            FN_t = np.sum(np.logical_and(np.equal(new_label, 1), np.equal(seg_t, 0)))

            TN_list[0][t] += TN_t
            TP_list[0][t] += TP_t
            FP_list[0][t] += FP_t
            FN_list[0][t] += FN_t

            # print(t1)

    np.seterr(divide='ignore', invalid='ignore')
    FPR_list[0] = (FP_list / (FP_list + TN_list))
    TPR_list[0] = (TP_list / (TP_list + FN_list))

def other_model_test(m_path, l_path, model1, model2, net_name1, net_name2):
    image_save_path = os.path.join(r'/home/wly/wly/data/Code/ALL_test', net_name2)
    if os.path.exists(image_save_path):
        print('ImageSavePath : {}  已存在\n'.format(image_save_path))
    else:
        os.mkdir(image_save_path)
        print('ImageSavePath : {}  创建成功\n'.format(image_save_path))

    # Coarse-network with up_sample
    seg_pre, _ = m_l_model_test(l_path, m_path, model1, net_name1)

    # Course-network without up_sample
    # seg_pre, _ = l_model_test( l_path,m_path, model1, net_name1)
    data_ = []
    label_pre = []

    for p in os.listdir(m_path):
        mr_image_path = os.path.join(m_path, p)
        mr_image = np.load(mr_image_path)
        mr_data = mr_image['original']
        mr_data = sktsf.resize(mr_data, (1024, 1024), mode='reflect', anti_aliasing=False)
        mr_label = mr_image['picc']
        mr_label = sktsf.resize(mr_label, (1024, 1024), mode='reflect', anti_aliasing=False)
        data_.append(mr_data)
        label_pre.append(mr_label)

    num = len(seg_pre)

    for i in range(0, num):
        seg_pre_ = seg_pre[i]
        seg_pre_ = sktsf.resize(seg_pre_, (1024, 1024), mode='reflect', anti_aliasing=False)
        data = data_[i]
        data = CLAHE(data)

        choice = np.where(label_pre[i] > 0.1)
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

        data_crop = data[x1:x2, y1:y2]
        seg_crop = seg_pre_[x1:x2, y1:y2]
        local_seg = local_test2(data_crop, seg_crop, model2, net_name2)
        seg1_with_seg2 = np.array(seg_pre_, copy=True)
        seg1_with_seg2[x1:x2, y1:y2] = local_seg[:, :]

        seg = sktsf.resize(seg1_with_seg2, (1024, 1024), mode='reflect', anti_aliasing=False)
        # ret, seg = cv2.threshold(seg, 0.1, 1, cv2.THRESH_BINARY)
        label = sktsf.resize(label_pre[i], (1024, 1024), mode='reflect', anti_aliasing=False)
        # ret, label = cv2.threshold(label, 0.1, 1, cv2.THRESH_BINARY)
        ret, new_label = cv2.threshold(label, 0.1, 1, cv2.THRESH_BINARY)

        TP_list = np.zeros((1, 100))
        FP_list = np.zeros((1, 100))
        FN_list = np.zeros((1, 100))
        TN_list = np.zeros((1, 100))


        for t in range(0, 100):
            t1 = t * 0.01
            ret, seg_t = cv2.threshold(seg, t1, 1, cv2.THRESH_BINARY)
            TN_t = np.sum(np.logical_and(np.equal(new_label, 0), np.equal(seg_t, 0)))
            TP_t = np.sum(np.logical_and(np.equal(new_label, 1), np.equal(seg_t, 1)))
            FP_t = np.sum(np.logical_and(np.equal(new_label, 0), np.equal(seg_t, 1)))
            FN_t = np.sum(np.logical_and(np.equal(new_label, 1), np.equal(seg_t, 0)))


            TN_list[0][t] += TN_t
            TP_list[0][t] += TP_t
            FP_list[0][t] += FP_t
            FN_list[0][t] += FN_t


            # print(t1)

    FPR_list[0] = (FP_list / (FP_list + TN_list))
    TPR_list[0] = (TP_list / (TP_list + FN_list))





if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 超参部分

    test_lr_path = r'/home/wly/wly/data/Code/pyramid_data/low_res/test/'
    test_mr_path = r'/home/wly/wly/data/Code/pyramid_data/mid_res/test/'

    from model.UNet_UP_Global import UNet as net1

    model1 = net1(1, 1, 32)
    net1_name = 'Global_Unet_UP_PPM_F5'

    from model.Unet_SW1_PAM_D import UNet

    from model.FCN_8s import FCN
    from model.CE_Net import CE_Net_
    from model.SegNet import SegNet
    from model.ResNet import Res_Net_

    # from model.Unet_SW1_PAM_D import UNet

    from model.Unet import UNet

    from new_Model.UNet_SW1 import UNet
    from new_Model.UNet_SW1_PAM import UNet
    from new_Model.UNet_SW1_PAM_D import UNet as net2

    from new_Model.UNet_SW2 import UNet
    from new_Model.UNet_SW2_PAM import UNet
    from new_Model.UNet_SW2_PAM_D import UNet

    from new_Model.UNet_SW3 import UNet
    from new_Model.UNet_SW3_PAM import UNet
    from new_Model.UNet_SW3_PAM_D import UNet

    # as net2
    # Local_Unet_F
    # SW1_13_F SW1_PAM_F1 SW1_PAM_D_F  SW1(k11)_PAM_F
    # SW2_F2  SW2_PAM_F SW2_PAM_D_F5
    # SW3_F  SW3_PAM_F SW3_PAM_D_F1
    # Local_Unet_F  Local_CE_NET_F  Local_SegNet_F  Local_FCN_F
    # Local_ResNet_F

    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline

    plt.figure()

    FPR_list = np.zeros((1, 100))
    TPR_list = np.zeros((1, 100))
    net2_name = 'SW1_PAM_D_F1'
    model2 = net2(2, 1)
    mid_res_local_test(test_mr_path, test_lr_path, model1, model2, net1_name, net2_name)

    FPR_list[0].sort()
    TPR_list[0].sort()

    # print(FPR_list)
    # print(TPR_list)

    # FPR_list[0][np.isnan(FPR_list[0])] = 0.5
    # TPR_list[0][np.isnan(TPR_list[0])] = 0.5
    f1 = np.polyfit(FPR_list[0], TPR_list[0], 2)
    p1 = np.poly1d(f1)
    yvals = p1(FPR_list[0])
    plt.plot(FPR_list[0], yvals, label='MAG-Net')
    # plt.plot(FPR_list[0], TPR_list[0], label='MAG-Net')

    FPR_list = np.zeros((1, 100))
    TPR_list = np.zeros((1, 100))
    net2_name = 'Local_Unet_F1'
    from model.Unet import UNet as net2
    model2 = net2(2, 1)
    other_model_test(test_mr_path, test_lr_path, model1, model2, net1_name, net2_name)
    FPR_list[0].sort()
    TPR_list[0].sort()

    f1 = np.polyfit(FPR_list[0], TPR_list[0], 2)
    p1 = np.poly1d(f1)
    yvals = p1(FPR_list[0])
    plt.plot(FPR_list[0], yvals, label='unet')
    # plt.plot(FPR_list[0], TPR_list[0],  label='unet')
    # #
    FPR_list = np.zeros((1, 100))
    TPR_list = np.zeros((1, 100))
    net2_name = 'Local_CE_NET_F1'
    from model.CE_Net import CE_Net_ as net2
    model2 = net2(2, 1)
    other_model_test(test_mr_path, test_lr_path, model1, model2, net1_name, net2_name)
    FPR_list[0].sort()
    TPR_list[0].sort()

    f1 = np.polyfit(FPR_list[0], TPR_list[0], 2)
    p1 = np.poly1d(f1)
    yvals = p1(FPR_list[0])
    plt.plot(FPR_list[0], yvals, label='cenet')
    # plt.plot(FPR_list[0], TPR_list[0], label='cenet')
    # #
    FPR_list = np.zeros((1, 100))
    TPR_list = np.zeros((1, 100))
    net2_name = 'Local_ResNet_F1'
    from model.ResNet import Res_Net_ as net2
    model2 = net2(2, 1)
    other_model_test(test_mr_path, test_lr_path, model1, model2, net1_name, net2_name)
    FPR_list[0].sort()
    TPR_list[0].sort()

    f1 = np.polyfit(FPR_list[0], TPR_list[0], 2)
    p1 = np.poly1d(f1)
    yvals = p1(FPR_list[0])
    plt.plot(FPR_list[0], yvals, label='resNet')
    # plt.plot(FPR_list[0], TPR_list[0], label='resNet')

    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1])
    plt.title('ROC')
    plt.xlabel('FPR_array')
    plt.ylabel('TPR_array')
    plt.legend()
    plt.show()






