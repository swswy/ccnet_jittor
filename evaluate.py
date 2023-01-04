import os
import jittor as jt
import utils as tools
from datasets.dataset import ADE20K
import numpy as np
from tqdm import tqdm
from networks.van_cca import CCNet_Model
# from networks.resnet_cca import CCNet_Model

BATCH_SIZE = 10
NUM_CLASSES = 151
VAL_DIR='.\ADEChallengeData2016'

# 对整个验证集进行计算
@jt.no_grad()
def valid_one_epoch(model, val_loader, criterion=None, epoch=0):
    model.eval()
    total_loss = 0
    acc = 0
    iou = 0
    miou = 0
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
    batch_num=len(val_loader)

    pbar = tqdm(val_loader)
    for idx, (images, labels) in enumerate(pbar):
        output = model(images)
        acc, iou, miou, hist = tools.get_MIoU(pred=output, label=labels, hist=hist)
        if criterion:
            loss = criterion(output, labels)
            total_loss += loss.data[0]

        pbar.set_description("(val)step[%d/%d]->loss:%.4f acc:%.4f miou:%.4f" %
                      (idx + 1, batch_num, total_loss/(idx + 1), acc, miou))

    epoch_loss = total_loss / batch_num
    epoch_acc = acc
    epoch_miou = miou

    # print("val->loss:%.4f acc:%.4f miou:%.4f" % (epoch_loss, acc, miou))
    with open("iou_eval.txt", "a") as f:
        f.write("epoch%d->"%(epoch) + str(iou) + "\n")
 
    # 保存hist矩阵
    # Hist_path = "./pic/epoch-%04d_val_hist.png"%(epoch)
    # tools.drawHist(hist, Hist_path)
 
    return epoch_loss, epoch_acc, epoch_miou



def eval(model,criterion=None, epoch=0):
    val_loader = ADE20K(dataset_dir=VAL_DIR, batch_size=BATCH_SIZE, mode='validation', shuffle=False)
    epoch_loss, epoch_acc, epoch_miou = valid_one_epoch(model, val_loader, criterion, epoch)
    return epoch_loss, epoch_acc, epoch_miou

def eval_all():
    model = CCNet_Model(num_classes=NUM_CLASSES,recurrence=2)
    list_dir = os.listdir("./checkpoint/")
    print("models include:",list_dir)
    max_miou = -1
    max_item = ""
    for idx,item in enumerate(list_dir):
        print(item)
        model.load("./checkpoint/"+item)
        epoch_loss, epoch_acc, epoch_miou = eval(model=model, epoch=idx)
        if(max_miou < epoch_miou):
            max_miou = epoch_miou
            max_item = item
    print("max miou:%.4f item:%s"%(max_miou, max_item))


if __name__ == "__main__":
    eval()
 