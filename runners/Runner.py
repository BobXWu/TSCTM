import torch
import numpy as np
from models.TSCTM import TSCTM


class Runner():
    def __init__(self, config, device):
        self.config = config
        self.model = TSCTM(config)
        self.model = self.model.to(device)

    def train(self, train_loader):
        data_size = len(train_loader.dataset)
        optimizer = torch.optim.Adam(self.model.parameters(), self.config.learning_rate)

        for epoch in range(1, self.config.num_epoch + 1):
            self.model.train()

            train_loss = 0
            contrastive_loss = 0
            for idx, bows in enumerate(train_loader):

                _, batch_loss, batch_contrastive_loss = self.model(bows)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                train_loss += batch_loss.cpu() * len(bows)
                contrastive_loss += batch_contrastive_loss.cpu() * len(bows)

            if epoch % 5 == 0:
                print('Epoch: {:03d}/{:03d} Loss: {:.3f}'.format(epoch, self.config.num_epoch, train_loss / data_size), end=' ')
                print('Contra_loss: {:.3f}'.format(contrastive_loss / data_size))

        beta = self.model.get_beta().detach().cpu().numpy()

        return beta

    def test(self, inputs):
        data_size = inputs.shape[0]
        theta = list()
        all_idx = torch.split(torch.arange(data_size), self.config.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_inputs = inputs[idx]
                batch_theta = self.model.get_theta(batch_inputs)
                theta += list(batch_theta.detach().cpu().numpy())

        theta = np.asarray(theta)
        return theta
