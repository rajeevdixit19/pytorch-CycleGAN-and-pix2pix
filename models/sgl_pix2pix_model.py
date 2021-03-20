import torch
from .base_model import BaseModel
from . import networks
from util.image_pool import ImagePool
from itertools import product
import random


class SglPix2PixModel(BaseModel):
    def name(self):
        return 'SglPix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.num_learners = 3
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        learner_number = [str(i) for i in range(1, self.num_learners + 1)]
        if self.isTrain:
            self.model_names = [''.join(list(i)) for i in product(['G', 'D'], learner_number)]
        else:  # during test time, only load Gs
            self.model_names = [''.join(list(i)) for i in product(['G'], learner_number)]
        # load/define networks
        self.netG = []
        for i in range(self.num_learners):
            self.netG.append(networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids))

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = []
            for i in range(self.num_learners):
                self.netD.append(networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids))

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = []
            self.optimizer_D = []
            for i in range(self.num_learners):
                self.optimizer_G.append(torch.optim.Adam(self.netG[i].parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999)))
                self.optimizer_D.append(torch.optim.Adam(self.netD[i].parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999)))

            self.optimizers.extend(self.optimizer_G)
            self.optimizers.extend(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = []
        for i in range(self.num_learners):
            self.fake_B.append(self.netG[i](self.real_A))

    def backward_D(self):
        temp = list(zip(self.netD, self.optimizer_D))
        random.shuffle(temp)
        self.netD, self.optimizer_D = zip(*temp)

        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.loss_D_fake = []

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.loss_D_real = []

        for i in range(self.num_learners):
            pred_fake = self.netD[i](fake_AB.detach())
            self.loss_D_fake.append(self.criterionGAN(pred_fake, False))

            pred_real = self.netD[i](real_AB)
            self.loss_D_real.append(self.criterionGAN(pred_real, True))

            # Combined loss
            self.loss_D = (self.loss_D_fake[i] + self.loss_D_real[i]) * 0.5
            self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        for i in range(self.num_learners):
            fake_AB = torch.cat((self.real_A, self.fake_B[i]), 1)
            pred_fake = self.netD[i](fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)

            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_B[i], self.real_B) * self.opt.lambda_A

            self.loss_G = self.loss_G_GAN + self.loss_G_L1

            self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        for optim in self.optimizer_D:
            optim.zero_grad()
        self.backward_D()
        for optim in self.optimizer_D:
            self.optimizer_D.step()

        for optim in self.optimizer_G:
            optim.zero_grad()
        self.backward_G()
        for optim in self.optimizer_G:
            optim.step()