import os
import jittor as jt
import jittor.nn as nn
from tqdm import tqdm
import argparse
from datasets.dataset import ADE20K
from networks.van_cca import CCNet_Model
# from networks.resnet_cca import CCNet_Model
from evaluate import eval
from tensorboardX import SummaryWriter
jt.flags.use_cuda = 1


@jt.enable_grad()
def train_one_epoch(model, train_loader, criterion, optimizer, epoch, opt, writer=None):
    model.train()
    losses = []
    batch_num = len(train_loader)

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]')
    for i, (images, labels) in enumerate(pbar):
        output = model(images)
        loss = criterion(output, labels)
        optimizer.step(loss)
        losses.append(loss.data[0])
        if writer:
            writer.add_scalar("loss", loss.data[0])

        pbar.set_description("Epoch-%s step[%d/%d]->loss:%.4f__lr-%.5f"%(epoch, i+1, batch_num, sum(losses) / len(losses), optimizer.lr))
        jt.sync_all()
        jt.gc()

        step = (batch_num * epoch + i + 1)
        if step % 5000 == 0:
            model.save(opt['save_dir'] + opt['save_name']+ "__{:0>5d}.pkl".format(step))

        update_lr(optimizer, step, opt['epochs'] * batch_num, opt['power'])

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def update_lr(optimizer, num, total, power):
    optimizer.lr = lr_poly(optimizer.lr, num, total, power)
    # for _, param_group in enumerate(optimizer.param_groups):
    #     if param_group.get("lr") != None:
    #         param_group["lr"] = lr_poly(param_group["lr"], num, total, power)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=2e-3, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--epochs', type=int, default=30, help="epochs")
    parser.add_argument('--ccatn_num', type=int, default=2, help="num of cc attention")
    parser.add_argument('--save_name',type=str,default="van_ccnet")

    args = parser.parse_args()
    options = {
        'lr': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'ccatn_num': args.ccatn_num,
        'save_name': args.save_name,
        
        'val': False,
        'power': 0.9,
        'momentum': 0.9,
        'num_classes': 151,
        'weight_decay': 5e-4,
        'log_dir': './log/',
        'save_dir': './checkpoint/',
        'dataset_dir': './ADEChallengeData2016/',
    }

    if not os.path.exists(options['save_dir']):
        os.makedirs(options['save_dir'])

    if not os.path.exists(options['log_dir']):
        os.makedirs(options['log_dir'])

    model = CCNet_Model(num_classes=options['num_classes'],recurrence=options['ccatn_num'])
    train_loader = ADE20K(dataset_dir=options['dataset_dir'], batch_size=options['batch_size'], mode='training', shuffle=True)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = nn.Adam(model.parameters(), options['lr'], weight_decay=options['weight_decay'])
    # optimizer = nn.SGD(model.parameters(), lr=options['lr'], momentum=options['momentum'], weight_decay=options['weight_decay'])
    
    best_val_miou=0
    best_epoch=0
    writer=SummaryWriter(log_dir=options['log_dir'])
    for epoch in range(options['epochs']):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, options, writer)

        if options['val'] and (epoch+1)%3==0:
            # 在验证集上计算
            val_loss, val_acc, val_miou = eval(model=model, criterion=criterion, epoch=epoch)
            infomation = "epoch-%02d__loss(val)-%.4f__acc-%.4f__miou(val)-%.4f" % (epoch, val_loss, val_acc, val_miou)
            print(infomation)
            if val_miou > best_val_miou:
                # 保存当前训练数据
                best_epoch=epoch
                best_val_miou=val_miou
                model.save(options['save_dir']+options['save_name']+"__00000.pkl")
            else:
                print("Validation moiu: %.4f, becomes smaller. Stop training." % val_miou)
                break

        # update_lr(optimizer, epoch, options['epochs'], options['power'])
        # print("learning rate has changed to", optimizer.lr)


