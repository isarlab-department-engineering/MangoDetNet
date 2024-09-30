import os
import glob
import time
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from tensorboardX import SummaryWriter


class BaseTrainer(metaclass=ABCMeta):

    def __init__(self, model, opt):

        self.model = model
        self.opt = opt
        self.writer_dir = ''
        self._trainloader=None
        self._validloader=None
        # self._testloader=None
        # self.check_config()
        
    @property
    def trainloader(self):
        return self._trainloader

    @trainloader.setter
    def trainloader(self, value):
        self._trainloader = value

    @property
    def validloader(self):
        return self._validloader

    @validloader.setter
    def validloader(self, value):
        self._validloader = value

    # @property
    # def testloader(self):
    #     return self._testloader
    #
    # @testloader.setter
    # def testloader(self, value):
    #     self._testloader = value

    def init_summary_writer(self):

        if self.opt['is_train']:
            self.writer_dir = os.path.join(self.opt['log_path'], self.opt['exp_name'], 'Train', 'events')
        else:
            self.writer_dir = os.path.join(self.opt['log_path'], self.opt['exp_name'], 'Test', 'events')

        if self.opt['start_epoch'] == 0:
            previous_events = glob.glob(os.path.join(self.writer_dir, '*'))
            for f in previous_events:
                os.remove(f)

        self.writer = SummaryWriter(log_dir=self.writer_dir)

    @abstractmethod
    def init_data_loader(self):
        pass

    def train(self):

        print("[*] Training starts...")

        if self.opt['resume_train']:
            total_steps = self.opt['start_epoch'] * len(self.trainloader) * self.opt['batch_size']
        else:
            total_steps = 0

        for epoch in range(self.opt['start_epoch'] + 1, self.opt['max_epochs'] + 1):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0

            epoch_losses = OrderedDict()
            losses = self.model.get_loss_names()
            for label in losses:
                epoch_losses[label] = 0.0
            for i, data in enumerate(self.trainloader, 0):

                iter_start_time = time.time()
                if total_steps % self.opt['print_freq'] == 0:
                    t_data = iter_start_time - iter_data_time

                total_steps += self.opt['batch_size']
                epoch_iter += self.opt['batch_size']

                self.model.set_input(data)
                self.model.optimize_parameters(epoch=epoch)

                losses = self.model.get_current_losses()
                for label, value in losses.items():
                    epoch_losses[label] += value

                if total_steps % self.opt['print_freq'] * self.opt['batch_size'] == 0:
                    t = (time.time() - iter_start_time) / self.opt['batch_size']
                    for label, value in losses.items():
                        self.writer.add_scalar('batch_loss/' + label, value, total_steps)
                        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
                        print(message)

                if total_steps % self.opt['print_hist_freq'] * self.opt['batch_size'] == 0:
                    histograms = self.model.get_current_histograms()
                    for name, param in histograms.items():
                        self.writer.add_histogram(name, param.clone().cpu().data.numpy(), total_steps)

                iter_data_time = time.time()

            for label, value in epoch_losses.items():
                self.writer.add_scalar('epoch/epoch_' + label, value / len(self.trainloader), epoch)

            if epoch % self.opt['save_epoch_freq'] == 0 and epoch != 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                self.model.save_networks(epoch)
                self.model.save_optimizers(epoch)
                self.model.set_train_mode()
            if epoch % self.opt['test_epoch_freq'] == 0:
                print('testing the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                self.test(epoch)
                self.model.set_train_mode()

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, self.opt['max_epochs'], time.time() - epoch_start_time))
            self.model.update_learning_rate()

    # @abstractmethod
    # def test(self, epoch=-1):
    #     pass
