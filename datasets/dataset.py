import os
from jittor.dataset import Dataset
from PIL import Image
from jittor import transform
import random
import numpy as np
import jittor as jt

# 超参数，设置裁剪的尺寸
CROP = 256

class ADE20K(Dataset):
    def __init__(self, dataset_dir, batch_size=16, mode='training', shuffle=False):
        super(ADE20K, self).__init__()
        self.dataset_dir = dataset_dir
        self.mode = mode # training | validation
        self.images_dir = os.path.join(self.dataset_dir, "images", mode)
        self.annotations_dir = os.path.join(self.dataset_dir, "annotations", mode)
        # length = 20210 if mode == 'training' else 2000
        length = 20000 if mode == 'training' else 2000
        self.set_attrs(
            batch_size=batch_size,
            total_len=length,
            shuffle=shuffle,
            # num_workers=2,
            drop_last=False,
        )
        self.img_tfs = transform.Compose([
            transform.ToTensor(),
            transform.ImageNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        image_name = 'ADE_train_{:0>8d}'.format(index+1) if self.mode=='training' else 'ADE_val_{:0>8d}'.format(index+1)
        label_path = os.path.join(self.annotations_dir, image_name+'.png')
        label = Image.open(label_path)
        image_path = os.path.join(self.images_dir, image_name+'.jpg')
        image = Image.open(image_path)
        image, label = self.img_process(image, label)
        return image, label

    def img_process(self, img:Image, img_gt:Image):
        #输入image对象，返回var对象
        if self.mode == "training":
            # 以50%的概率左右翻转
            if random.random() > 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                img_gt = img_gt.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            # 以50%的概率上下翻转
            if random.random() > 0.5:
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                img_gt = img_gt.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            # 以50%的概率像素矩阵转置
            if random.random() > 0.5:
                img = img.transpose(Image.Transpose.TRANSPOSE)
                img_gt = img_gt.transpose(Image.Transpose.TRANSPOSE)
            # 进行随机裁剪
            width, height = img.size
            st = random.randint(0,20)
            box = (st, st, width-1, height-1)
            img = img.crop(box)
            img_gt = img_gt.crop(box)

        img = img.resize((CROP, CROP))
        img_gt = img_gt.resize((CROP, CROP))

        img = self.img_tfs(img)
        img_gt = np.array(img_gt)
        img_gt = jt.Var(img_gt)

        return img, img_gt

if __name__ == "__main__":
    train_loader = ADE20K(dataset_dir='./ADEChallengeData2016/', batch_size=2, mode='training', shuffle=True)
    for images,labels in train_loader:
        print(images.shape)
        print(labels.shape)
        break             