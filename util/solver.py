import os
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torchvision.utils as tvutil

class Solver(object):
    default_adam_args = {"lr": 1e-3,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self,
                 optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.PoissonNLLLoss(log_input=False, full=True),
                 l1_w=1e-5,
                 l2_w=1e-5,
                 log_folder='logs'):
        """
        Optimizer of models.
        :param optim: Optimizer algorithm (default: Adam optimizer)
        :param optim_args: Parameters of the optimizer
        :param loss_func: Loss function (default: Negative log likelihood of Poisson - PoissonNLLLoss)
        :param l1_w: L1 regularization weight
        :param l2_w: L2 regularization weight
        """
        self.log_folder = log_folder
        os.makedirs(log_folder, exist_ok=True)
        self.logger = SummaryWriter(log_folder)

        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.l1_w = l1_w
        self.l2_w = l2_w

    def _reset_histories(self):
        """
        Resets the loss history of the model.
        """
        self.train_loss_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=1, mdl_file='model'):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: validation data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: Report loss every n iteration
        - mdl_file: file name of saved model
        """
        self.log_weights(model)
        optim = self.optim(model.parameters(), **self.optim_args)
        self.logger.add_text('Model architecture', repr(model))
        self.logger.add_text('Optimizer', repr(optim))
        self._reset_histories()

        if torch.cuda.is_available():
            model.cuda()

        print('Start training...')

        for epoch in range(num_epochs):
            for i, (inputs, targets) in enumerate(train_loader, 1):
                inputs, targets = inputs.float(), targets.float()

                if model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                optim.zero_grad()
                outputs = model.forward(inputs)

                # Regularizer (TO DO later: Reg should be moved to model class because it is model dependent and hinders
                # the solver class to be general for all models

                l1_regularizer = self.l1_w * torch.norm(model.fc.weight.view(-1), 1)
                l2_regularizer = self.l2_w * (torch.norm(model.conv1.weight.view(-1), 2) + torch.norm(model.conv2.weight.view(-1), 2))

                loss = self.loss_func(outputs, torch.squeeze(targets, dim=1)) + l1_regularizer + l2_regularizer

                loss.backward()

                optim.step()

                print('[Epoch %d/%d] [Batch %d/%d] [Train loss: %f]' % (epoch, num_epochs,  i, len(train_loader),
                                                                        loss.data.cpu().numpy()))

                self.train_loss_history.append(loss.data.cpu().numpy())
                batches_done = epoch * len(train_loader) + i
                self.logger.add_scalar('train_loss', self.train_loss_history[-1], batches_done)
                if batches_done % log_nth == 0:
                    val_loss = self.test(model, val_loader)
                    self.val_loss_history.append(val_loss)
                    print('[Epoch %d/%d] [Batch %d/%d] [Val loss: %f]' % (epoch, num_epochs, i, len(train_loader), val_loss))
                    self.logger.add_scalar('val_loss', self.train_loss_history[-1], batches_done)

            model.save(self.log_folder + '//' + mdl_file + '.mdl')
        self.logger.close()
        print('FINISH.')

    def log_weights(self, model):
        conv1_filter = model.conv1.weight.detach()
        conv1_img = tvutil.make_grid(conv1_filter.view(conv1_filter.size(0)*conv1_filter.size(1),
                                                       1, conv1_filter.size(2), conv1_filter.size(2)),
                                     nrow=conv1_filter.size(1),
                                     normalize=True, padding=3)
        self.logger.add_image('conv1_filters', conv1_img)

        conv2_filter = model.conv2.weight.detach()
        conv2_img = tvutil.make_grid(conv2_filter.view(conv2_filter.size(0) * conv2_filter.size(1),
                                                       1, conv2_filter.size(2), conv2_filter.size(2)),
                                     nrow=conv2_filter.size(1),
                                     normalize=True, padding=3)
        self.logger.add_image('conv2_filters', conv2_img)

        filtsz = model.l3_filt_shape
        fc_filter = model.fc.weight.view(filtsz[0]*filtsz[1], 1, filtsz[2], filtsz[3])
        fc_img = tvutil.make_grid(fc_filter, nrow=53, normalize=True)
        self.logger.add_image('fc_filters', fc_img)

    def test(self, model, val_loader):
        model.eval()  # Set model state to evaluation
        val_losses = []
        for j, (inputs, targets) in enumerate(val_loader, 1):
            inputs = Variable(inputs.float())
            targets = Variable(targets.float())
            if model.is_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model.forward(inputs)
            loss = self.loss_func(outputs, torch.squeeze(targets, dim=1))
            val_losses.append(loss.data.cpu().numpy())
        model.train()
        return np.mean(val_losses)
