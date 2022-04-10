from tkinter.messagebox import NO
import torch
import numpy as np
import random
import argparse
import os
import os.path as osp
import time
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ae import Autoencoder
from rbm import RBM, RBMHandle

def set_seeds(torch_seed, cuda_seed, np_seed, random_seed):
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(cuda_seed)
    np.random.seed(np_seed)
    random.seed(random_seed)

def is_zero(x):
    return torch.abs(x) < 1e-6

def nan_assert(x):
    assert torch.any(torch.isnan(x)) == False

def gpu_state(gpu_id, get_return=False):
    qargs = ['index', 'gpu_name', 'memory.used', 'memory.total']
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))

    results = os.popen(cmd).readlines()
    gpu_id_list = gpu_id.split(",")
    gpu_space_available = {}
    for cur_state in results:
        cur_state = cur_state.strip().split(", ")
        for i in gpu_id_list:
            if i == cur_state[0]:
                if not get_return:
                    print(f'GPU {i} {cur_state[1]}: Memory-Usage {cur_state[2]} / {cur_state[3]}.')
                else:
                    gpu_space_available[i] = int("".join(list(filter(str.isdigit, cur_state[3])))) - int("".join(list(filter(str.isdigit, cur_state[2]))))
    if get_return:
        return gpu_space_available

def set_gpu(x, space_hold=0):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    gpu_available = 0
    while gpu_available < space_hold:
        gpu_space_available = gpu_state(x, get_return=True)
        for gpu_id, space in gpu_space_available.items():
            gpu_available += space
        if gpu_available < space_hold:
            gpu_available = 0
            time.sleep(1800) # 间隔30分钟.
    gpu_state(x)


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--init_weights', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--lr_rbm', type=float, default=0.001)
    parser.add_argument('--lr_ae', type=float, default=0.0001)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--loss_fn', type=str, default='nn-mse')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    parser.add_argument('--do_train_rbm', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--bonus', action='store_true', default=False)

    parser.add_argument('--time_str', type=str, default='')
    parser.add_argument('--notes', type=str, default='')

    args = parser.parse_args()

    return args

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
IMAGE_PATH = '/data/zhangyk/data/debias'

class LfFColoredMNIST(Dataset):
    def __init__(self, dataset_name, split, train_corr, severity, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        skewed_str = f'Skewed{1 - train_corr:.3}' if train_corr != 1.0 else 'Skewed0'
        self.data = np.load(os.path.join(IMAGE_PATH, f'{dataset_name}-{skewed_str}-Severity{severity}/{split}/images.npy'))
        attrs = np.load(os.path.join(IMAGE_PATH, f'{dataset_name}-{skewed_str}-Severity{severity}/{split}/attrs.npy'))
        torch_attrs = torch.from_numpy(attrs)
        self.targets = torch_attrs[:, 0]
        self.biased_targets = torch_attrs[:, 1]

        if split != 'test':
            # 数据集全shuffle, 类别的label和bias label都是乱的.
            indices = np.arange(len(self.data))
            np.random.shuffle(indices)
            self.data = self.data[indices]
            self.targets = self.targets[indices]
            self.biased_targets = self.biased_targets[indices]

            # self.confusion_matrix_org, self.confusion_matrix, self.confusion_matrix_by = get_confusion_matrix(
            #     num_classes=10,
            #     targets=self.targets,
            #     biases=self.biased_targets)

            conflict_data, conflict_targets, conflict_biased_targets = [], [], []
            for i in torch.sort(self.targets.unique())[0]:
                cur_idx = (self.targets == i) & (self.biased_targets != i)
                if torch.sum(cur_idx).item() != 0:
                    conflict_data.append(self.data[cur_idx])
                    conflict_targets.append(self.targets[cur_idx])
                    conflict_biased_targets.append(self.biased_targets[cur_idx])

            if len(conflict_data) > 0 and len(conflict_targets) > 0 and len(conflict_biased_targets) > 0:
                self.conflict_data = np.concatenate(conflict_data)
                self.conflict_targets = torch.cat(conflict_targets)
                self.conflict_biased_targets = torch.cat(conflict_biased_targets)
                if self.conflict_data.shape[0] > 10000:
                    overmuch_sampled_idx = np.random.choice(self.conflict_data.shape[0], 10000, replace=False)
                    self.conflict_data = self.conflict_data[overmuch_sampled_idx]
                    self.conflict_targets = self.conflict_targets[overmuch_sampled_idx]
                    self.conflict_biased_targets = self.conflict_biased_targets[overmuch_sampled_idx]

                img_list, target_list, bias_list = [], [], []

                for img, target, bias in zip(self.conflict_data, self.conflict_targets, self.conflict_biased_targets):
                    target, bias = int(target), int(bias)
                    img = Image.fromarray(img.astype(np.uint8), mode='RGB')
                    # img.save(f'imgs_conflicting/{int(target)}_{int(bias)}_{random.randint(0, 10000)}.jpg')

                    if self.transform is not None:
                        img = self.transform(img)
                    if self.target_transform is not None:
                        target = self.target_transform(target)

                    img_list.append(img)
                    target_list.append(target)
                    bias_list.append(bias)

                if isinstance(img_list[0], list):
                    img_list = [torch.cat([i[0], i[1]]) for i in img_list]
                self.conflict_data, self.conflict_targets, self.conflict_biased_targets = torch.stack(img_list), torch.LongTensor(target_list), torch.LongTensor(bias_list)

    def __getitem__(self, index):
        img, target, bias = self.data[index], int(self.targets[index]), int(self.biased_targets[index])
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')
        # img.save(f'imgs_aligned/{target}_{bias}_{random.randint(0, 10000)}.jpg')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target, bias, index
        return img, (target, bias)

    def __len__(self):
        return len(self.data)


def get_colored_mnist_dataloader(batch_size, data_label_correlation, severity, split='train', num_workers=0):
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    dataset = LfFColoredMNIST(
        dataset_name='ColoredMNIST',
        split=split,
        train_corr=data_label_correlation,
        severity=severity,
        transform=train_transform
        )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return dataloader


class PrepareFunc(object):
    def __init__(self, args):
        self.args = args

    def load_rbm_pretrained_models(self, ae_model, rbm_models):
        rbm_model_length = len(rbm_models)
        if rbm_model_length == 0:
            return

        model_dict = ae_model.state_dict()
        pretrained_dict = {}
        for i, cur_model in enumerate(rbm_models):
            encoder_id = i
            decoder_id = rbm_model_length - 1 - i
            pretrained_dict[f'encoder.{encoder_id}.0.weight'] = "TODO"
            pretrained_dict["TODO"] = cur_model.bias_h
            pretrained_dict[f'decoder.{decoder_id}.0.weight'] = "TODO"
            pretrained_dict["TODO"] = "TODO"

        model_dict.update(pretrained_dict)
        ae_model.load_state_dict(model_dict)

    def prepare_model(self, model_name, feature_dims=None):
        if model_name == 'rbm_handle':
            return RBMHandle()

        if model_name == 'rbm':
            model = RBM(feature_dims[0], feature_dims[1], self.args.k)
        elif model_name == 'ae':
            model = Autoencoder(layers=feature_dims)

        # # load pre-trained model (no FC weights 注意只加载backbone的参数.)
        # if self.args.init_weights is not None:
        #     print(f'Loading the pre-training model...')

        #     def load_weights(cur_model):
        #         model_dict = cur_model.state_dict()

        #         if osp.isfile(self.args.init_weights):
        #             pretrained_dict = torch.load(self.args.init_weights)['model']
        #         else:
        #             raise Exception('Loading pretrained model error: file not exists.')

        #         def fine_tuning_params_dict(cur_d, modify_str=None, option='flt'):
        #             # 为了对应 model_dict.keys(), 微调一下pretrained模型的key.
        #             if option == 'add':
        #                 ret_d = {modify_str + k: v for k, v in cur_d.items() if modify_str + k in model_dict}
        #                 del_keys = [k for k in cur_d.keys() if modify_str + k not in model_dict]
        #             elif option == 'del':
        #                 ret_d = {k.replace(modify_str, ''): v for k, v in cur_d.items() if k.replace(modify_str, '') in model_dict}
        #                 del_keys = [k for k in cur_d.keys() if k.replace(modify_str, '') not in model_dict]
        #             elif option == 'flt':
        #                 ret_d = {k: v for k, v in cur_d.items() if k in model_dict}
        #                 del_keys = [k for k in cur_d.keys() if k not in model_dict]
        #             if len(del_keys) != 0:
        #                 print(f'The deleted pre-trained\'s keys (IN pre-trained model BUT NOT IN current): {del_keys}.')

        #             return ret_d

        #         def substr_in_keys(cur_d, cur_s):
        #             # 只要存在一个有就return True.
        #             for i in cur_d.keys():
        #                 if cur_s in i:
        #                     return True
        #             return False

        #         if len(pretrained_dict.keys()) == 0:
        #             raise Exception('Oops! Failed to load pre-trained model.')
        #         elif len(pretrained_dict.keys()) == len(model_dict.keys()):
        #             print('All pre-trained model parameters are loaded.')
        #         else:
        #             print(f'Missing loading pre-training model parameters (IN current model BUT NOT IN pre-trained) {[i for i in model_dict.keys() if i not in pretrained_dict.keys()]}.')

        #         model_dict.update(pretrained_dict)
        #         cur_model.load_state_dict(model_dict)

        #     # load 核心函数结束.
        #     load_weights(model)
        #     # check_model_changed_handle.set_model(model)

        model = model.to(torch.device('cuda'))

        return model

    def prepare_dataloader(self, dataset_name):
        if dataset_name == 'MNIST':
            train_dataset = torchvision.datasets.MNIST(root=self.args.data_dir, train=True, transform=torchvision.transforms.ToTensor(), download=True)
            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers)
            test_dataset = torchvision.datasets.MNIST(root=self.args.data_dir, train=False, transform=torchvision.transforms.ToTensor(), download=True)
            test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers)
        
        elif dataset_name == 'ColoredMNIST':
            train_loader = get_colored_mnist_dataloader(
                batch_size=self.args.batch_size,
                data_label_correlation=0.9,
                severity=1,
                split='train',
                num_workers=self.args.num_workers
                )
            test_loader = get_colored_mnist_dataloader(
                batch_size=self.args.batch_size,
                data_label_correlation=0.9,
                severity=1,
                split='valid',
                num_workers=self.args.num_workers
                )

        return train_loader, test_loader

    def prepare_loss_fn(self):
        if self.args.loss_fn == 'nn-mse':
            return nn.MSELoss()

    def prepare_optimizer(self, model):
        def set_optimizer(optimizer_name, cur_encoder):
            if optimizer_name == 'adam':
                return optim.Adam(
                    model.parameters(),
                    lr=self.args.lr_ae,
                    weight_decay=self.args.weight_decay
                    )

        optimizer = set_optimizer(self.args.optimizer, model)

        def set_lr_scheduler(cur_type, optmz):
            if cur_type == 'step':
                return optim.lr_scheduler.StepLR(
                    optmz,
                    step_size=int(self.args.step_size),
                    gamma=self.args.gamma
                    )
            elif cur_type == 'multistep':
                return optim.lr_scheduler.MultiStepLR(
                    optmz,
                    milestones=[int(_) for _ in self.args.step_size.split(',')],
                    gamma=self.args.gamma,
                    )
            elif cur_type == 'cosine':
                return optim.lr_scheduler.CosineAnnealingLR(
                    optmz,
                    self.args.max_epoch,
                    eta_min=self.args.cosine_annealing_lr_eta_min   # a tuning parameter
                    )
            elif cur_type == 'plateau':
                return optim.lr_scheduler.ReduceLROnPlateau(
                    optmz,
                    mode='min',
                    factor=self.args.gamma,
                    patience=5
                    )
            else:
                raise ValueError('No Such Scheduler')

        # lr_scheduler = set_lr_scheduler(self.args.lr_scheduler, optimizer)

        return optimizer

def cal_debias_al_acc(preds_idx, Y, Yb):
    result = []
    pred_bool_idx = torch.zeros(Y.shape[0], dtype=torch.bool)
    pred_bool_idx[preds_idx] = True
    if Y.unique().shape[0] != 1:
        for i in torch.sort(Y.unique())[0]:
            conflict_in_class_idx = (Y == i) & (Yb != i)
            result.append((conflict_in_class_idx & pred_bool_idx).sum().item() / conflict_in_class_idx.sum().item())

    conflict_idx = (Y != Yb)
    result.append((conflict_idx & pred_bool_idx).sum().item() / conflict_idx.sum().item())
    return result

def debias_dataloader2tensor(model, cur_loader, category=None, is_raw=False):
    model.eval()
    with torch.no_grad():
        hidden, label, label_bias, query_label = [], [], [], []
        for _, (x, (y, y_bias)) in enumerate(cur_loader):
            if x.shape[1] == 3:
                x = x.mean(dim=1, keepdim=True)
            if category is not None:
                cur_idx = (y == category)
                if cur_idx.sum() == 0:
                    continue
                x, y, y_bias = x[cur_idx], y[cur_idx], y_bias[cur_idx]

            x = x.view(x.shape[0], -1).to(torch.device('cuda'))
            if is_raw:
                hidden.append(x.detach().cpu())
            else:
                hidden.append(model.encoder(x).detach().cpu())
            label.append(y.cpu())
            label_bias.append(y_bias.cpu())

            query_label.append((y != y_bias).long().cpu())

        return torch.cat(hidden), torch.cat(label), torch.cat(label_bias), torch.cat(query_label)