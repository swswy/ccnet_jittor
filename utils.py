import numpy as np
# import matplotlib.pyplot as plt


NUM_CLASSES = 151

def fast_hist(a,b,n):
    '''
    a是预测值, b是ground truth, ab形状(h x w), n是类别数
    '''
    a=a.numpy()
    b=b.numpy()
    a = np.argmax(a, axis=1) #(NUM_CLASSES,h x w)->(h x w,)
    k = (b >= 0) & (b < n) # k是一个一维bool数组，形状(h×w,)；目的是找出标签中需要计算的类别（去掉背景
    return np.bincount(a[k].astype(int) + n * b[k].astype(int), minlength=n**2).reshape(n, n)

def per_class_iou(hist):
    '''
    分别为每个类别计算mIoU, hist的形状(n, n)
    '''
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def per_class_acc(hist):

    return np.diag(hist) / np.maximum(hist.sum(1), 1) # maximum是处理分母为零的情况
    # np.seterr(divide="ignore", invalid="ignore")
    # acc_cls = np.diag(hist) / hist.sum(1)
    # np.seterr(divide="warn", invalid="warn")
    # acc_cls[np.isnan(acc_cls)] = 0.
    # return acc_cls


def get_MIoU(pred, label, hist):
    """
    :param pred: 预测向量
    :param label: 真实标签值
    :return: 准确率, 每类的准确率. 每类的iou, miou, 混淆矩阵
    """
    hist = hist + fast_hist(pred, label, NUM_CLASSES)
    # 准确率
    acc = np.diag(hist).sum() / hist.sum()
    # 每类的准确率
    # acc_cls = per_class_acc(hist)
    # 每类的iou
    iou = per_class_iou(hist)
    miou = np.nanmean(iou[1:])
    return acc, iou, miou, hist

# def drawHist(hist, path):
#     # print(hist)
#     hist_ = hist[1:]
#     hist_tmp = np.zeros((NUM_CLASSES-1, NUM_CLASSES-1))
    
#     for i in range(len(hist_)):
#         hist_tmp[i] = hist_[i][1:]

#     # print(hist_tmp)
#     hist = hist_tmp
#     plt.matshow(hist)
#     plt.xlabel("Predicted label")
#     plt.ylabel("True label")
#     plt.axis("off")
#     plt.colorbar()
#     plt.show()
#     if(path != None):
#         plt.savefig(path)
#         print("%s保存成功✿✿ヽ(°▽°)ノ✿"%path)


if __name__ == "__main__":
    # hist = np.random.randint(0,20,size=(151,151))
    # drawHist(hist, None)
    image_gt = np.random.randint(0,4,(1,4,4))
    image = np.random.rand(1,4,4,4)
    print(image_gt)
    print(image)
    hist=fast_hist(image, image_gt, 4)
    print(hist.shape)
    print(hist)
    iou = per_class_iou(hist)
    miou = np.nanmean(iou[1:])
    print("iou",iou)
    print("miou",miou)
