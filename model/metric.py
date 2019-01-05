import torch
import numpy as np

def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def acc_somecls(output, target, cls_ind):
    # accuracy on class 1
    with torch.no_grad():
        pred    = torch.argmax(output, dim=1)
        assert pred.shape[0]    == len(target)
        pred_arr    = pred.cpu().numpy()
        target_arr  = target.cpu().numpy()
        i_index = np.where(target_arr == cls_ind)[0]
        i_correct   = np.sum(pred_arr[i_index] == cls_ind) / (float(len(i_index)) + 0.00001)
    return i_correct

def acc_cls1(output, target):
    return acc_somecls(output, target, cls_ind=0)
def acc_cls2(output, target):
    return acc_somecls(output, target, cls_ind=1)
def acc_cls3(output, target):
    return acc_somecls(output, target, cls_ind=2)
def acc_cls4(output, target):
    return acc_somecls(output, target, cls_ind=3)
def acc_cls5(output, target):
    return acc_somecls(output, target, cls_ind=4)
def acc_cls6(output, target):
    return acc_somecls(output, target, cls_ind=5)
def acc_cls7(output, target):
    return acc_somecls(output, target, cls_ind=6)
def acc_cls8(output, target):
    return acc_somecls(output, target, cls_ind=7)
def acc_cls9(output, target):
    return acc_somecls(output, target, cls_ind=8)
def acc_cls10(output, target):
    return acc_somecls(output, target, cls_ind=9)
def acc_cls11(output, target):
    return acc_somecls(output, target, cls_ind=10)
def acc_cls12(output, target):
    return acc_somecls(output, target, cls_ind=11)
def acc_cls13(output, target):
    return acc_somecls(output, target, cls_ind=12)
def acc_cls14(output, target):
    return acc_somecls(output, target, cls_ind=13)
def acc_cls15(output, target):
    return acc_somecls(output, target, cls_ind=14)
def acc_cls16(output, target):
    return acc_somecls(output, target, cls_ind=15)
def acc_cls17(output, target):
    return acc_somecls(output, target, cls_ind=16)
def acc_cls18(output, target):
    return acc_somecls(output, target, cls_ind=17)
#acc_cls1    = lambda output, target: acc_somecls(output, target, cls_ind=0)
#acc_cls2    = lambda output, target: acc_somecls(output, target, cls_ind=1)
#acc_cls3    = lambda output, target: acc_somecls(output, target, cls_ind=2)
#acc_cls4    = lambda output, target: acc_somecls(output, target, cls_ind=3)
#acc_cls5    = lambda output, target: acc_somecls(output, target, cls_ind=4)
#acc_cls6    = lambda output, target: acc_somecls(output, target, cls_ind=5)
#acc_cls7    = lambda output, target: acc_somecls(output, target, cls_ind=6)
#acc_cls8    = lambda output, target: acc_somecls(output, target, cls_ind=7)
#acc_cls9    = lambda output, target: acc_somecls(output, target, cls_ind=8)
#acc_cls10   = lambda output, target: acc_somecls(output, target, cls_ind=9)
#acc_cls11   = lambda output, target: acc_somecls(output, target, cls_ind=10)
#acc_cls12   = lambda output, target: acc_somecls(output, target, cls_ind=11)
#acc_cls13   = lambda output, target: acc_somecls(output, target, cls_ind=12)
#acc_cls14   = lambda output, target: acc_somecls(output, target, cls_ind=13)
#acc_cls15   = lambda output, target: acc_somecls(output, target, cls_ind=14)
#acc_cls16   = lambda output, target: acc_somecls(output, target, cls_ind=15)
#acc_cls17   = lambda output, target: acc_somecls(output, target, cls_ind=16)
#acc_cls18   = lambda output, target: acc_somecls(output, target, cls_ind=17)

def accuracy_metric_eachclass(output, target):
    with torch.no_grad():
        pred    = torch.argmax(output, dim=1)
        assert pred.shape[0]    == len(target)
        class_tuple = tuple(set(target.tolist()))
        class_num   = len(class_tuple)
        pred_arr    = pred.cpu().numpy()
        target_arr  = target.cpu().numpy()
        correct_arr = np.zeros((class_num, ))
        for i in range(class_num):
            i_index = np.where(target_arr == i)[0]
            i_correct   = np.sum(pred_arr[i_index] == i) / (float(len(i_index)) + 0.00001)
            correct_arr[i]  = i_correct
        print(correct_arr)
    return correct_arr

def accuracy_metric(output, target):
    with torch.no_grad():
        pred    = torch.argmax(output, dim=1)
        assert pred.shape[0]    == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def accuracy_top3_metric(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def accuracy_top5_metric(output, target, k=5):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
