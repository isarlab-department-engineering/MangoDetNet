from abc import ABCMeta, abstractmethod
import torch
import os
from torch.optim import lr_scheduler
from collections import OrderedDict
from torch import nn
import math


class BaseModel(metaclass=ABCMeta):

    def __init__(self, opt, log_path):
        self.opt = opt
        self.log_path = log_path
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.histograms_names = []
        self.image_paths = []
        self.save_dir = ''
        self.optimizers = OrderedDict()

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def set_input(self, inputs):
        pass

    @abstractmethod
    def optimize_parameters(self, epoch):
        pass

    @abstractmethod
    def initialize(self, opt):
        pass

    # @abstractmethod
    # def log_visuals(self, writer, total_steps):
    #     pass


    def get_scheduler(self, optimizer):

        if self.opt['resume_train']:
            last_epoch = self.opt['start_epoch']
        else:
            last_epoch = -1

        scheduler = lr_scheduler.StepLR(optimizer=optimizer,
                                        step_size=self.opt['lr_decay_iters'],
                                        gamma=self.opt['lr_step_gamma'],
                                        last_epoch=last_epoch)

        return scheduler

    # load and print networks; create schedulers
    def setup(self):

        if self.opt['is_train']:
            self.save_dir = os.path.join(self.log_path, self.opt['exp_name'], 'Train')
            if self.opt['resume_train']:
                self.load_optimizers(self.opt['start_epoch'])
            else:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            self.schedulers = [self.get_scheduler(optimizer) for (name, optimizer) in self.optimizers.items()]
        else:
            self.save_dir = os.path.join(self.log_path, self.opt['exp_name'], 'Test')

        if not self.opt['is_train'] or self.opt['resume_train'] or not self.opt['start_epoch'] == 0:
            self.load_networks(self.opt['start_epoch'])

        self.print_networks(True)
        if self.opt['is_train']:
            self.save_config()

    # make models eval mode during test time
    def set_eval_mode(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()


    def set_train_mode(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def val(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    # update learning rate (called once every epoch)
    def update_learning_rate(self, iter=None):
        for scheduler in self.schedulers:
            scheduler.step(epoch=iter)
        for (name, optimizer) in (self.optimizers.items()):
            print('learning rate for optimizer %s: %.7f' % (name, optimizer.param_groups[0]['lr']))
        # lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate = %.7f' % lr)

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                tensor_cpu = getattr(self, name).clone().detach().cpu()
                visual_ret[name] = tensor_cpu
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_histograms(self):
        histograms_ret = OrderedDict()
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                for name, param in net.named_parameters():
                    if name in self.histograms_names:
                        histograms_ret[name] = param
        return histograms_ret

    def get_loss_names(self):
        names = []
        for name in self.loss_names:
            names.append(name)
        return names

    # save models to the disk
    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, 'weights')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_file = os.path.join(save_path, save_filename)
                net = getattr(self, 'net' + name)
                torch.save(net.state_dict(), save_file)

    # save models to the disk
    def save_optimizers(self, which_epoch):
        for (name, optimizer) in self.optimizers.items():
            save_filename = '%s_optimizer_%s.pth' % (which_epoch, name)
            save_path = os.path.join(self.save_dir, 'weights')
            save_file = os.path.join(save_path, save_filename)
            torch.save(optimizer.state_dict(), save_file)

    # load models from the disk
    def load_networks(self, which_epoch):
        for name in self.model_names:

            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.log_path, self.opt['exp_name'], 'Train', 'weights', load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)

                device = 'cuda:' + str(self.opt['gpu_id'])
                checkpoint = torch.load(load_path, map_location=device)
                net.load_state_dict(checkpoint)

    # load models from the disk
    def load_optimizers(self, which_epoch):
        for (name, optimizer) in self.optimizers.items():
            load_filename = '%s_optimizer_%s.pth' % (which_epoch, name)
            load_path = os.path.join(self.save_dir, 'weights', load_filename)
            print('loading the optimizer from %s' % load_path)
            checkpoint = torch.load(load_path)
            optimizer.load_state_dict(checkpoint)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        dir = os.path.join(self.save_dir, 'config')
        if not os.path.exists(dir):
            os.makedirs(dir)
        file = os.path.join(dir, 'network.txt')
        with open(file, 'w') as f:
            for name in self.model_names:
                if isinstance(name, str):
                    net = getattr(self, 'net' + name)
                    num_params = 0
                    for param in net.parameters():
                        num_params += param.numel()
                    f.write(str(net))
                    f.write('\n')
                    if verbose:
                        print(net)
                    print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
            f.close()
        print('-----------------------------------------------')


    def save_config(self):

        dir = os.path.join(self.save_dir, 'config')
        if not os.path.exists(dir):
            os.makedirs(dir)
        file = os.path.join(dir, 'configuration.txt')
        with open(file, 'w') as f:
            for key, value in self.opt.items():
                f.write(key + ': ' + str(value))
                f.write('\n')
            f.close()

    # set requies_grad=False to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def initialize_weights(self, network):
        for m in network.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
