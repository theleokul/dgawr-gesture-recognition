import torch
from util.DHG_parse_data import *
from util.Mydataset import *
import torch.optim as optim
import numpy as np
from datetime import datetime
import time
import argparse
import os
from model.network import *
from model.network_awr import PoseNet
from model.feature_tool import FeatureModule
from loss import My_SmoothL1Loss
import random
from tqdm import tqdm
from PIL import Image



parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, default=32)  # 16
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')  # 1000

parser.add_argument('--patiences', default=50, type=int,
                    help='number of epochs to tolerate no improvement of val_loss')  # 1000


parser.add_argument('--test_subject_id', type=int, default=3,
                    help='id of test subject, for cross-validation')

parser.add_argument('--data_cfg', type=int, default=0,
                    help='0 for 14 class, 1 for 28')


parser.add_argument('--dp_rate', type=float, default=0.2,
                    help='dropout rate')  # 1000


img_size = 128
downsample = 2
kernel_size = 0.4
time_len = 8
# time_len_replacements = 3
ft_sz = int(img_size / downsample)
FM = FeatureModule()



def init_data_loader(test_subject_id, data_cfg):

    train_data, test_data = get_train_test_data(test_subject_id, data_cfg)


    # TODO: use_data_aug = False -> to train key points detection
    train_dataset = Hand_Dataset(train_data, use_data_aug = False, time_len = time_len)

    test_dataset = Hand_Dataset(test_data, use_data_aug = False, time_len = time_len)

    print("train data num: ",len(train_dataset))
    print("test data num: ",len(test_dataset))

    print("batch size:", args.batch_size)
    print("workers:", args.workers)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    return train_loader, val_loader


def init_model(data_cfg):
    if data_cfg == 0:
        class_num = 14
    elif data_cfg == 1:
        class_num = 28

    model = DG_STA(class_num, args.dp_rate)
    model = torch.nn.DataParallel(model).cuda()
    
    model_detection = PoseNet('hourglass_1', joint_num=22)
    model_detection = torch.nn.DataParallel(model_detection).cuda()

    return model, model_detection


def depth_topil(depth, uvd, b=0, t=0, r=1):
    x = np.repeat(depth[b, t, 0].cpu().numpy()[:, :, None], axis=2, repeats=3)
    x = (((x + 1) / 2) * 255).astype(np.uint8)
    
    uvd = ((uvd + 1) / 2 * x.shape[0]).long()
    
    uvd = uvd.long()
    for i in range(22):
        x[uvd[b, t, i, 1] - r : uvd[b, t, i, 1] + r, uvd[b, t, i, 0] - r : uvd[b, t, i, 0] + r] = np.array([255, 0, 0])
    
    return Image.fromarray(x)


def model_forward(sample_batched, model, model_detection, criterion, criterion_detection):

    depth = sample_batched["depth"].cuda().float()
    jt_uvd_gt = sample_batched["skeleton_proj"].cuda().float()  # 32, 8, 22, 2
    
    label = sample_batched["label"]
    label = label.type(torch.LongTensor)
    label = label.cuda()
    label = torch.autograd.Variable(label, requires_grad=False)

    # Select random time slot for training detection
    ts = random.choice(list(range(time_len)))
    depth_ts = depth[:, ts]
    offset_gt_ts = FM.joint2offset(jt_uvd_gt[:, ts], depth[:, ts], kernel_size, ft_sz).cuda()
    offset_pred_ts = model_detection(depth_ts)    
    loss_detection = criterion_detection(offset_pred_ts, offset_gt_ts)
    # loss_detection = 0
    
    # Inference all the coords for classification pipeline
    # jt_uvd_pred = []
    # for ts in range(time_len):
    #     depth_ts = depth[:, ts]
    #     offset_pred_ts = model_detection(depth_ts).detach()
    #     jt_uvd_pred_ts = FM.offset2joint_softmax(offset_pred_ts, depth_ts, kernel_size).cuda()
        
    #     # TODO: Scale coordinates back to original scale (broken after resampling to 128x128)
    #     # jt_uvd_pred_ts[:, 0] *= sample_batched["mult_w"].cuda().unsqueeze(1)
    #     # jt_uvd_pred_ts[:, 1] *= sample_batched["mult_h"].cuda().unsqueeze(1)
    
    #     jt_uvd_pred.append(jt_uvd_pred_ts)
            
    # jt_uvd_pred = torch.stack(jt_uvd_pred, dim=1).detach()
    
    # score = model(jt_uvd_pred)
    score = model(jt_uvd_gt)

    # loss_clf = criterion(score, label)
    loss_clf = 0

    acc = get_acc(score, label)

    return score, loss_detection, loss_clf, acc


def get_acc(score, labels):
    score = score.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    outputs = np.argmax(score, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    print("\nhyperparamter......")
    args = parser.parse_args()
    print(args)

    print("test_subject_id: ", args.test_subject_id)

    #folder for saving trained model...
    # change this path to the fold where you want to save your pre-trained model
    model_fold = "/home/l.kulikov/dev/DGAWR/exp_detection_only/DHS_ID-{}_dp-{}_lr-{}_dc-{}/".format(args.test_subject_id,args.dp_rate, args.learning_rate, args.data_cfg)
    os.makedirs(model_fold, exist_ok=True)

    train_loader, val_loader = init_data_loader(args.test_subject_id,args.data_cfg)


    #.........inital model
    print("\ninit model.............")
    model, model_detection = init_model(args.data_cfg)
    model_solver = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    model_detection_solver = optim.Adam(filter(lambda p: p.requires_grad, model_detection.parameters()), lr=args.learning_rate)

    #........set loss
    criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion_detection = My_SmoothL1Loss().cuda()

    #
    train_data_num = 2660
    test_data_num = 140
    iter_per_epoch = int(train_data_num / args.batch_size)

    #parameters recording training log
    max_acc = 0
    min_val_loss = np.inf
    no_improve_epoch = 0
    n_iter = 0

    #***********training#***********
    for epoch in range(args.epochs):
        print("\ntraining.............")
        model.train()
        start_time = time.time()
        train_acc = 0
        train_loss = 0
        train_loss_detection = 0
        train_loss_clf = 0
        
        
        pbar = tqdm(
            enumerate(train_loader), 
            leave=True, 
            total=len(train_loader), 
            desc=f'epoch: {epoch}',
        )
        for i, sample_batched in pbar:
            n_iter += 1
            #print("training i:",i)
            if i + 1 > iter_per_epoch:
                continue
            # score,loss, acc = model_forward(sample_batched, model, criterion)
            score, loss_detection, loss_clf, acc = model_forward(sample_batched, model, model_detection, criterion, criterion_detection)
            # loss = 10_000 * loss_detection + loss_clf
            loss = loss_detection

            # model.zero_grad()
            model_detection.zero_grad()
            
            loss.backward()
            
            # model_solver.step()
            model_detection_solver.step()

            train_acc += acc
            train_loss += loss
            train_loss_detection += loss_detection
            train_loss_clf += loss_clf
            
            pbar.set_postfix({
                'loss_detection': 10_000 * train_loss_detection.item() / float(i + 1),
                # 'loss_clf': train_loss_clf.item() / float(i + 1),
                'acc': train_acc / float(i + 1)
            })

        train_acc /= float(i + 1)
        train_loss /= float(i + 1)

        print("*** DHS  Epoch: [%2d] time: %4.4f, "
              "cls_loss: %.4f  train_ACC: %.6f ***"
              % (epoch + 1,  time.time() - start_time,
                 train_loss.data, train_acc))
        start_time = time.time()

        #adjust_learning_rate(model_solver, epoch + 1, args)
        #print(print(model.module.encoder.gcn_network[0].edg_weight))

        #***********evaluation***********
        with torch.no_grad():
            val_loss = 0
            acc_sum = 0
            model.eval()
            for i, sample_batched in enumerate(val_loader):
                #print("testing i:", i)
                label = sample_batched["label"]
                score, loss_detection, loss_clf, acc = model_forward(sample_batched, model, model_detection, criterion, criterion_detection)
                # loss = 10_000 * loss_detection + loss_clf
                loss = loss_detection
                val_loss += loss

                if i == 0:
                    score_list = score
                    label_list = label
                else:
                    score_list = torch.cat((score_list, score), 0)
                    label_list = torch.cat((label_list, label), 0)


            val_loss = val_loss / float(i + 1)
            val_cc = get_acc(score_list,label_list)


            print("*** DHS  Epoch: [%2d], "
                  "val_loss: %.6f,"
                  "val_ACC: %.6f ***"
                  % (epoch + 1, val_loss, val_cc))

            #save best model
            # if val_cc > max_acc:
            #     max_acc = val_cc
            #     no_improve_epoch = 0
            #     val_cc = round(val_cc, 10)

            #     ckpt = {
            #         'model': model.state_dict(),
            #         'model_detection': model_detection.state_dict()
            #     }
                
            #     torch.save(
            #         ckpt,
            #         '{}/epoch_{}_acc_{}.pth'.format(model_fold, epoch + 1, val_cc)
            #     )
            #     print("performance improve, saved the new model......best acc: {}".format(max_acc))
            # else:
            #     no_improve_epoch += 1
            #     print("no_improve_epoch: {} best acc {}".format(no_improve_epoch,max_acc))
            
            #save best model
            val_loss = val_loss.item()
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                no_improve_epoch = 0
                val_loss = round(val_loss, 10)

                ckpt = {
                    'model': model.state_dict(),
                    'model_detection': model_detection.state_dict()
                }
                
                torch.save(
                    ckpt,
                    '{}/epoch_{}_acc_{}.pth'.format(model_fold, epoch + 1, val_loss)
                )
                print("performance improve, saved the new model......min loss: {}".format(val_loss))
            else:
                no_improve_epoch += 1
                print("no_improve_epoch: {} min loss {}".format(no_improve_epoch, min_val_loss))

            if no_improve_epoch > args.patiences:
                print("stop training....")
                break
