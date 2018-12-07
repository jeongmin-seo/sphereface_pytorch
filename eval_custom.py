from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import numpy as np

from custom_dataloader import FaceDataLoader
from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere
import torch._utils

torch.backends.cudnn.bencmark = True

# for old version pytorch model
try:
    torch._utils._rebuild_tensor_v2

except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

parser = argparse.ArgumentParser(description='PyTorch sphereface for custom data')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--data_root', default='./data/code (2)/deep_learning_data_x', type=str)
parser.add_argument('--model_path', default='/home/mlpa/ssd/face_recognition/sphereface_pytorch/model/sphere20a_19.pth',type=str)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', default=256, type=int, help='')
parser.add_argument('--epoch', default=1000, type=int, help='train epoch')
args = parser.parse_args()


def cosine_distance(f1, f2):
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-5)
    # return  f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)


def set_fine_tune_model(_model_path, _n_class):
    _model = net_sphere.sphere20a()
    state_dict = torch.load(_model_path)
    _model.load_state_dict(state_dict)
    in_feature = _model.fc6.in_features
    _model.fc6 = net_sphere.AngleLinear(in_feature, _n_class)

    """
    for i, param in enumerate(_model.parameters()):
        if i < 2:
            param.requires_grad = False
    """
    return _model


def test_model_load(_model_path):
    _model = net_sphere.sphere20a(feature=True)
    state_dict = torch.load(_model_path)
    _model.load_state_dict(state_dict)

    return _model


def fine_tune(_trian_loader, _model, n_epoch):

    loss_func = net_sphere.AngleLoss().cuda()
    optimizer = torch.optim.SGD(_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    _model.cuda()
    _model.train()
    for epoch in range(1, n_epoch+1):
        print(epoch)
        for i, (data, labels) in enumerate(_trian_loader):
            input_var = Variable(data, volatile=True).cuda(async=True)
            target_var = Variable(labels, volatile=True).cuda().long()

            output = _model(input_var)
            loss = loss_func(output, target_var)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return _model


def save_gallery(_loader, _test_model):
    gallery = dict()

    _test_model.cuda()
    for i, (data, labels) in enumerate(_loader):

        images = Variable(data, volatile=True).cuda(async=True)
        features = _test_model(images)
        features = features.data.cpu().numpy()

        for i, label in enumerate(labels):
            if not label in list(gallery.keys()):
                gallery[label] = [features[i]]  # TODO: feature to cpu version
                continue

            gallery[label].append(features[i])

    return gallery

def calc_max_similarity(train_features, test_features):

    max_sim = None
    for i, tr_feat in enumerate(train_features):
        for j, te_feat in enumerate(test_features):
            cur_sim = cosine_distance(tr_feat, te_feat)
            max_sim = check_max(cur_sim, max_sim)

    return max_sim


def check_max(cur, max):
    if not max or max < cur:
        return cur

    elif max > cur:
        return max


def calc_accuracy(pred, true):
    assert len(pred) == len(true)
    return (np.asarray(pred) == np.asarray(true)).sum()/len(true)


def eval_using_gallery(train_gallery, test_gallery):

    prediction = list()
    true_label = list()
    for test_key in list(test_gallery.keys()):
        cur_similarity = None
        cur_result = None
        for train_key in list(train_gallery.keys()):
            calc_sim = calc_max_similarity(train_gallery[train_key], test_gallery[test_key])

            if not cur_similarity or cur_similarity < calc_sim:
                cur_result = train_key
                cur_similarity = calc_sim

        true_label.append(test_key)
        prediction.append(cur_result)

    print(calc_accuracy(prediction, true_label))
    return prediction, true_label


if __name__=="__main__":
    global args
    args = parser.parse_args()

    data_loader = FaceDataLoader(batch_size=args.bs, num_workers=4, path=args.data_root, txt_path="./")
    train_loader, test_loader = data_loader.run()

    """
    model = set_fine_tune_model(args.model_path, 2000)
    fine_tune(train_loader, model, args.epoch)
    """

    test_model = test_model_load(args.model_path)
    train_gallery = save_gallery(train_loader, test_model)
    test_gallery = save_gallery(test_loader, test_model)

    print(list(train_gallery.keys()))
    print(list(test_gallery.keys()))

    eval_using_gallery(train_gallery, test_gallery)
