import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.WILDCATpooling import WildcatPool2d, ClassWisePool
from models.BaseModel import BaseModel
from models.kornia_ssim_loss import SSIMLoss


def define_pooling():
    pooling = nn.Sequential()
    pooling.add_module('class_wise', ClassWisePool(num_maps=3))
    pooling.add_module('spatial', WildcatPool2d(kmax=0.2, kmin=None, alpha=0.7))
    spatial_pooling = pooling
    return spatial_pooling

def predict_conv(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 3, kernel_size=3, padding=1),
        # nn.Sigmoid()
    )

def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, output_padding=0),
        nn.ReLU(inplace=True)
    )

def blackconv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0),
        nn.ReLU(inplace=True)
    )

def up_decod(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

def decod(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
    )

def crop_like(input, ref):
    assert (input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]


class EncoderDecoderModel(BaseModel):

    def name(self):
        return 'EncoderDecoderModel'

    def initialize(self, opt):

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['SSIMLoss', 'MSELoss'] if self.opt['strategy'] == 'Reconstruction' else ['BCELoss']

        # specify the models you want to save to the disk.
        # The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['resnet50',
                            'upconv6', 'upconv5', 'upconv4', 'upconv3', 'upconv2', 'upconv1',
                            'up_decod6', 'up_decod5', 'up_decod4', 'up_decod3', 'up_decod2', 'up_decod1',
                            'blackconv6', 'blackconv5', 'blackconv4', 'blackconv3', 'blackconv2', 'blackconv1',
                            'decod'
                            ]

        self.spatial_pooling = define_pooling()
        self.netresnet50 = models.resnet50(pretrained=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.netupconv6 = upconv(2048, 1024)
        self.netupconv5 = upconv(512, 512)
        self.netupconv4 = upconv(256, 256)
        self.netupconv3 = upconv(128, 128)
        self.netupconv2 = upconv(64, 64)
        self.netupconv1 = upconv(32, 32)

        self.netup_decod6 = up_decod(1024 + 1024, 512)
        self.netup_decod5 = up_decod(512 + 512, 256)
        self.netup_decod4 = up_decod(256 + 256, 128)
        self.netup_decod3 = up_decod(128, 64)
        self.netup_decod2 = up_decod(64, 32)
        self.netup_decod1 = up_decod(32, 16)

        self.netblackconv6 = blackconv(2048, 512)
        self.netblackconv5 = blackconv(512, 256)
        self.netblackconv4 = blackconv(256, 128)
        self.netblackconv3 = blackconv(128, 64)
        self.netblackconv2 = blackconv(64, 32)
        self.netblackconv1 = blackconv(32, 16)

        self.netdecod = decod(16, 3)

        if self.opt['is_train']:

            # define loss functions
            self.criterion_SSIM = SSIMLoss(11)
            self.criterion_MSE = nn.MSELoss()
            self.criterion_BCE = nn.BCEWithLogitsLoss()

            # initialize optimizers
            self.optimizer_resnet50 = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                                             self.netresnet50.parameters()),
                                                      lr=opt['init_lr'] * 0.1)

            self.optimizers_upconvs = torch.optim.SGD(list(self.netupconv6.parameters()) +
                                                      list(self.netupconv5.parameters()) +
                                                      list(self.netupconv4.parameters()) +
                                                      list(self.netupconv3.parameters()) +
                                                      list(self.netupconv2.parameters()) +
                                                      list(self.netupconv1.parameters()), lr=opt['init_lr'])

            self.optimizers_up_decod = torch.optim.SGD(list(self.netup_decod6.parameters()) +
                                                       list(self.netup_decod5.parameters()) +
                                                       list(self.netup_decod4.parameters()) +
                                                       list(self.netup_decod3.parameters()) +
                                                       list(self.netup_decod2.parameters()) +
                                                       list(self.netup_decod1.parameters()), lr=opt['init_lr'])

            self.optimizers_blackconv = torch.optim.SGD(list(self.netblackconv6.parameters()) +
                                                        list(self.netblackconv5.parameters()) +
                                                        list(self.netblackconv4.parameters()) +
                                                        list(self.netblackconv3.parameters()) +
                                                        list(self.netblackconv2.parameters()) +
                                                        list(self.netblackconv1.parameters()), lr=opt['init_lr'])

            self.optimizers_decod = torch.optim.SGD(list(self.netdecod.parameters()), lr=opt['init_lr'])

            self.optimizers['resnet50'] = self.optimizer_resnet50
            self.optimizers['UpConvs'] = self.optimizers_upconvs
            self.optimizers['UpDecod'] = self.optimizers_up_decod
            self.optimizers['BlackConv'] = self.optimizers_blackconv
            self.optimizers['Decod'] = self.optimizers_decod

        if self.opt['gpu_id'] >= 0 and torch.cuda.is_available():

            self.netresnet50.cuda()

            self.netupconv6.cuda()
            self.netupconv5.cuda()
            self.netupconv4.cuda()
            self.netupconv3.cuda()
            self.netupconv2.cuda()
            self.netupconv1.cuda()

            self.netup_decod6.cuda()
            self.netup_decod5.cuda()
            self.netup_decod4.cuda()
            self.netup_decod3.cuda()
            self.netup_decod2.cuda()
            self.netup_decod1.cuda()

            self.netblackconv6.cuda()
            self.netblackconv5.cuda()
            self.netblackconv4.cuda()
            self.netblackconv3.cuda()
            self.netblackconv2.cuda()
            self.netblackconv1.cuda()

            self.netdecod.cuda()

    def set_input(self, inputs):
        self.input = inputs[0]
        self.labels = inputs[1]
        if self.opt['gpu_id'] >= 0 and torch.cuda.is_available():
            self.input = self.input.cuda()
            self.labels = self.labels.cuda()

    def backward(self, retain_graph=False):

        if self.opt['strategy'] == 'Reconstruction':
            loss_SSIM = self.criterion_SSIM(self.decoded_img, self.input)
            loss_MSE = self.criterion_MSE(self.decoded_img, self.input)

            loss_SSIM.backward(retain_graph=retain_graph)
            loss_MSE.backward(retain_graph=retain_graph)

            self.loss_SSIMLoss = loss_SSIM
            self.loss_MSELoss = loss_MSE

        else:
            loss_BCE = self.criterion_BCE(self.prediction, self.labels)

            loss_BCE.backward(retain_graph=retain_graph)

            self.loss_BCELoss = loss_BCE

    def val_loss(self):

        if self.opt['strategy'] == 'Reconstruction':
            loss_SSIM = self.criterion_SSIM(self.decoded_img, self.input)
            loss_MSE = self.criterion_MSE(self.decoded_img, self.input)

            self.loss_SSIMLoss = loss_SSIM
            self.loss_MSELoss = loss_MSE
        else:
            loss_BCE = self.criterion_BCE(self.prediction, self.labels)

            self.loss_BCELoss = loss_BCE

    def forward(self):

        out_conv1 = self.netresnet50.conv1(self.input)
        out_conv1 = self.netresnet50.bn1(out_conv1)
        out_conv1 = self.netresnet50.relu(out_conv1)
        out_conv2 = self.netresnet50.maxpool(out_conv1)
        out_conv3 = self.netresnet50.layer1(out_conv2)

        # pooling between first and second resnet block needed to predict output at 4 scales
        out_conv3 = self.max_pool(out_conv3)
        out_conv4 = self.netresnet50.layer2(out_conv3)
        out_conv5 = self.netresnet50.layer3(out_conv4)
        out_conv6 = self.netresnet50.layer4(out_conv5)

        upsample6 = self.netupconv6(out_conv6)
        upsample6black = self.netblackconv6(
            F.interpolate(out_conv6, scale_factor=2, mode='bilinear', align_corners=False))
        upsample6 = torch.cat((upsample6, out_conv5), 1)
        out_up_decod6 = self.netup_decod6(upsample6)

        upsample5 = self.netupconv5(torch.add(out_up_decod6, upsample6black))
        upsample5black = self.netblackconv5(
            F.interpolate(out_up_decod6, scale_factor=2, mode='bilinear', align_corners=False))
        upsample5 = torch.cat((crop_like(upsample5, out_conv4), out_conv4), 1)
        out_up_decod5 = self.netup_decod5(upsample5)

        upsample4 = self.netupconv4(torch.add(out_up_decod5, crop_like(upsample5black, out_up_decod5)))
        upsample4black = self.netblackconv4(
            F.interpolate(out_up_decod5, scale_factor=2, mode='bilinear', align_corners=False))
        upsample4 = torch.cat((crop_like(upsample4, out_conv3), out_conv3), 1)
        out_up_decod4 = self.netup_decod4(upsample4)

        upsample3 = self.netupconv3(torch.add(out_up_decod4, crop_like(upsample4black, out_up_decod4)))
        upsample3black = self.netblackconv3(
            F.interpolate(out_up_decod4, scale_factor=2, mode='bilinear', align_corners=False))
        out_up_decod3 = self.netup_decod3(upsample3)

        upsample2 = self.netupconv2(torch.add(out_up_decod3, upsample3black))
        upsample2black = self.netblackconv2(
            F.interpolate(out_up_decod3, scale_factor=2, mode='bilinear', align_corners=False))
        out_up_decod2 = self.netup_decod2(upsample2)

        upsample1 = self.netupconv1(torch.add(out_up_decod2, upsample2black))
        upsample1black = self.netblackconv1(
            F.interpolate(out_up_decod2, scale_factor=2, mode='bilinear', align_corners=False))
        out_up_decod1 = self.netup_decod1(upsample1)

        self.decoded_img = self.netdecod((crop_like(torch.add(out_up_decod1, upsample1black), self.input)))

        if self.opt['strategy'] != 'Reconstruction':
            self.prediction = self.spatial_pooling(self.decoded_img)

        return self.decoded_img

    def inference_forward(self):
        return self.forward()

    def optimize_parameters(self, epoch):

        print('Optimizing')
        self.set_requires_grad([self.netresnet50,
                                self.netupconv6, self.netupconv5, self.netupconv4,
                                self.netupconv3, self.netupconv2, self.netupconv1,
                                self.netup_decod6, self.netup_decod5, self.netup_decod4,
                                self.netup_decod3, self.netup_decod2, self.netup_decod1,
                                self.netblackconv6, self.netblackconv5, self.netblackconv4,
                                self.netblackconv3, self.netblackconv2, self.netblackconv1,
                                self.netdecod
                                ], True)

        self.optimizer_resnet50.zero_grad()
        self.optimizers_upconvs.zero_grad()
        self.optimizers_up_decod.zero_grad()
        self.optimizers_decod.zero_grad()
        self.optimizers_blackconv.zero_grad()

        self.forward()
        self.backward(retain_graph=True)

        self.optimizer_resnet50.step()
        self.optimizers_upconvs.step()
        self.optimizers_up_decod.step()
        self.optimizers_decod.step()
        self.optimizers_blackconv.step()
