import numpy as np
import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-3,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self,
                 optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.PoissonNLLLoss(log_input=False, full=True),
                 l1_w=1e-5,
                 l2_w=1e-5):
        """
        Optimizer of models.
        :param optim: Optimizer algorithm (default: Adam optimizer)
        :param optim_args: Parameters of the optimizer
        :param loss_func: Loss function (default: Negative log likelihood of Poisson - PoissonNLLLoss)
        :param l1_w: L1 regularization weight
        :param l2_w: L2 regularization weight
        """

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

        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        nIterations = num_epochs * iter_per_epoch
        it = 0
        print('Start training...')

        epoch_dict = {}
        for epoch in range(num_epochs):

            for i, (inputs, targets) in enumerate(train_loader, 1):
                inputs, targets = inputs.float(), targets.float()

                if model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                optim.zero_grad()
                outputs = model.forward(inputs)

                # Regularizer (TO DO later: Reg should be moved to model class because it is model dependent and hinders
                # the solver class to be general for all models
                conv1_param = torch.cat([x.view(-1)
                                         for x in model.conv1.parameters()])
                conv2_param = torch.cat([x.view(-1)
                                         for x in model.conv2.parameters()])
                fc_param = torch.cat([x.view(-1)
                                      for x in model.fc.parameters()])

                l1_regularizer = self.l1_w * torch.norm(fc_param, 1)
                l2_regularizer = self.l2_w * (torch.norm(conv1_param, 2) + torch.norm(conv2_param, 2))

                loss = self.loss_func(outputs, targets) + l1_regularizer + l2_regularizer

                loss.backward()

                optim.step()

                print('[Iteration %i/%i] Train loss: %f' % (it, nIterations,
                                                            loss.data.cpu().numpy()))

                self.train_loss_history.append(loss.data.cpu().numpy())

                if it % log_nth == 0:
                    val_loss = self.test(model, val_loader)
                    self.val_loss_history.append(val_loss)
                    print('[Iteration %i/%i] Validation loss: %f' % (it, nIterations, val_loss))

                it += 1

            model.save(mdl_file + '.mdl')

        print('FINISH.')

    def test(self, model, val_loader):
        model.eval()  # Set model state to evaluation
        val_losses = []
        for j, (inputs, targets) in enumerate(val_loader, 1):
            inputs = Variable(inputs.float())
            targets = Variable(targets.float())
            if model.is_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model.forward(torch.squeeze(inputs, 0))
            loss = self.loss_func(outputs, targets)
            val_losses.append(loss.data.cpu().numpy())
        model.train()
        return np.mean(val_losses)
