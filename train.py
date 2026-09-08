# coding=utf-8
import os
import time
import shutil

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from common import FineGrainedDataset, load_hierarchy_info, resnet50, seed_everything, set_gpu_envs

#####################################################
setname = 'CUB_H'  # 'CUB_H', 'Butterfly_H'
CUDA_VISIBLE_DEVICES = [2, 3]
batch_size = 32
base_lr = 0.006
input_size = 448  # 224, 448

#####################################################
pretrained = True
transfer_mode = 'random'
transfer_mode_test = 'center'

model_name = 'resnet-50-coarse-fine-size-'
model_name += str(input_size) + '-hier'
model_path = './experiments/' + setname

##########################info about the training############################
end_epoch = 90
num_workers = 4
seed_everything(seed=65)

# set the initial learning rate
init_lr = base_lr * batch_size / 64.0
lrf = min(1e-4, init_lr * 0.01)
weight_decay = 1e-4

warmup_epoch = 20
lr_step_gamma = [30, 60]


def main():
    gpu_envs_str = set_gpu_envs(CUDA_VISIBLE_DEVICES)
    time_str = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(model_path, model_name)
    lr = init_lr
    softmax = torch.nn.Softmax(dim=1)
    codepath = os.path.join(save_path, "code-{}".format(time_str))
    if os.path.exists(codepath):
        shutil.rmtree(codepath)
    os.makedirs(codepath)

    trainset = FineGrainedDataset(setname, input_size, transfer_mode, is_train=True, show_info=True)
    testset = FineGrainedDataset(setname, input_size, transfer_mode_test, is_train=False, show_info=True)

    hierarchy_info = load_hierarchy_info(setname)
    label_len = hierarchy_info['label_len']

    # get the mapping matrix between label granularities
    trans_1_to_2 = torch.from_numpy(hierarchy_info['trans_1_to_2']).type(torch.float32)
    trans_2_to_3 = torch.from_numpy(hierarchy_info['trans_2_to_3']).type(torch.float32)
    trans_3_to_4 = torch.from_numpy(hierarchy_info['trans_3_to_4']).type(torch.float32)
    trans_1_to_4 = torch.matmul(trans_1_to_2, torch.matmul(trans_2_to_3, trans_3_to_4)).cuda()
    trans_2_to_4 = torch.matmul(trans_2_to_3, trans_3_to_4).cuda()
    trans_3_to_4 = trans_3_to_4.cuda()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               pin_memory=True, drop_last=False, prefetch_factor=2, persistent_workers=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                              pin_memory=True, drop_last=False, prefetch_factor=2, persistent_workers=True)

    num_classes = label_len
    model = resnet50(pretrained=pretrained, num_classes=num_classes)
    print(num_classes)

    def test(model, testloader, criterion):
        model.eval()
        raw_correct = np.zeros([5])
        with torch.no_grad():
            for i, data in enumerate(tqdm(testloader)):
                images, labels = data
                images, labels = images.cuda(), labels.type(torch.long).cuda()
                outputs_list = model(images)

                # correct num
                one_merged = torch.zeros_like(outputs_list[-1])
                for subidx in range(len(outputs_list)):
                    pred = outputs_list[subidx].max(1, keepdim=True)[1]
                    raw_correct[subidx] += pred.eq(labels[:, subidx].view_as(pred)).sum().item()
                one_merged += torch.matmul(softmax(outputs_list[0]), trans_1_to_4)
                one_merged += torch.matmul(softmax(outputs_list[1]), trans_2_to_4)
                one_merged += torch.matmul(softmax(outputs_list[2]), trans_3_to_4)
                one_merged += softmax(outputs_list[-1])
                pred = one_merged.max(1, keepdim=True)[1]
                raw_correct[-1] += pred.eq(labels[:, -1].view_as(pred)).sum().item()
        raw_accuracy = raw_correct / len(testloader.dataset)
        return raw_accuracy

    acc_list = []
    loss_list = []
    start_epoch = 0

    ################check the start_epoch###############
    assert start_epoch < end_epoch

    ################ Set visible GPU IDs; virtual IDs start from 0 ###############
    print('GPU available:', gpu_envs_str)
    print('Data mode:', transfer_mode)
    vitual_gpu_id = [i for i in range(len(CUDA_VISIBLE_DEVICES))]
    model = torch.nn.DataParallel(model, device_ids=vitual_gpu_id)
    model.cuda()

    ######################## training setup #######################
    criterion = nn.CrossEntropyLoss().cuda()

    ######################## optimizer setup #######################
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, end_epoch, eta_min=lrf, last_epoch=-1)

    time_str = time.strftime("%Y%m%d-%H%M%S")
    acc_save_file = open(os.path.join(codepath, time_str + '-acc-list.txt'), 'w')

    # record the acc and loss changes
    acc_array_save = acc_list
    loss_array_save = loss_list

    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.cuda()
        model.train()
        lr = next(iter(optimizer.param_groups))['lr']
        train_loss_avg = 0
        for i, data in enumerate(tqdm(trainloader)):
            images, labels = data
            images, labels = images.cuda(), labels.type(torch.long).cuda()

            outputs_list = model(images)

            loss_1 = criterion(outputs_list[0], labels[:, 0])
            loss_2 = criterion(outputs_list[1], labels[:, 1])
            loss_3 = criterion(outputs_list[2], labels[:, 2])
            loss_4 = criterion(outputs_list[3], labels[:, 3])

            total_loss = loss_1 + loss_2 + loss_3 + loss_4
            train_loss_avg += total_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        scheduler.step()
        train_loss_avg /= len(trainloader)

        # eval testset
        raw_acc_list = test(model, testloader, criterion)

        save_one_epoch_str = '[{:d}/{:d}] Test raw accuracy: {:.2f}%,{:.2f}%,{:.2f}%,{:.2f}%, fused: {:.2f}%, loss: {:.3f}'.format(
            epoch, end_epoch, 100. * raw_acc_list[0], 100. * raw_acc_list[1], 100. * raw_acc_list[2],
            100. * raw_acc_list[3], 100. * raw_acc_list[4], train_loss_avg)
        print(save_one_epoch_str)

        acc_array_save.append(raw_acc_list)
        loss_array_save.append(train_loss_avg.detach().cpu().numpy())
        acc_save_file.write(save_one_epoch_str + '\n')

        torch.save({'epoch': epoch,
                    'model_state_dict': model.cpu().module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.cpu().state_dict(),
                    'learning_rate': lr,
                    'acc_array_save': acc_array_save,
                    'loss_array_save': loss_array_save},
                   os.path.join(codepath, 'last-model.pth'))
    acc_save_file.close()


if __name__ == '__main__':
    # Run one training job per dataset. Edit the globals below (or add another
    # block) to sweep additional datasets / hyperparameters.

    ##########################info about the training############################
    setname = 'Butterfly_H'  # 'CUB_H', 'Butterfly_H'
    CUDA_VISIBLE_DEVICES = [1, 3]
    batch_size = 64
    base_lr = 0.010
    input_size = 224  # 224, 448

    #####################################################
    pretrained = True
    transfer_mode = 'random'
    transfer_mode_test = 'center'

    model_name = 'resnet-50-coarse-fine-size-'
    model_name += str(input_size) + '-hier'
    model_path = './experiments/' + setname

    end_epoch = 90
    num_workers = 4
    #####################################################
    main()

    ##########################info about the training############################
    setname = 'CUB_H'  # 'CUB_H', 'Butterfly_H'
    CUDA_VISIBLE_DEVICES = [1, 3]
    batch_size = 32
    base_lr = 0.010
    input_size = 448  # 224, 448

    #####################################################
    pretrained = True
    transfer_mode = 'random'
    transfer_mode_test = 'center'

    model_name = 'resnet-50-coarse-fine-size-'
    model_name += str(input_size) + '-hier'
    model_path = './experiments/' + setname

    end_epoch = 20
    num_workers = 4
    #####################################################
    main()
