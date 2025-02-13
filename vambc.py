import torch
import torch.nn as nn


class Vambc(nn.Module):
    def __init__(self, dim_x, dim_lstm, dim_z, zb_std, num_clusters):
        super(Vambc, self).__init__()
        self.encoder = nn.LSTM(dim_x, dim_lstm, batch_first=True)
        self.decoder = nn.LSTM(dim_z, dim_lstm, proj_size=dim_z, batch_first=True)
        self.output = nn.Linear(dim_z, dim_x)

        self.fc_y = nn.Linear(dim_lstm, num_clusters)
        self.fc_cluster = nn.Linear(num_clusters, dim_z)
        self.fc_zb = nn.Linear(dim_lstm, dim_z)

        self.zb_mean = nn.Parameter(torch.zeros(dim_z), requires_grad=True)
        self.zb_std = zb_std # constant standard deviation
        self.num_clusters = num_clusters

    def lstm_decode(self, init, num_steps):
        """
        init: shape (batch_size, dim_z)
        """
        decodings = []
        for step in range(num_steps):
            if step == 0:
                o, (h, c) = self.decoder(init.unsqueeze(1))
            else:
                o, (h, c) = self.decoder(h.squeeze().unsqueeze(1), (h, c))
            decodings.append(self.output(o))
        return torch.concatenate(decodings, dim=1)

    def forward(self, x):
        """Forward pass

        Args:
            x (tensor): shape (batch_size, seq_len, dim_x)

        Returns:
            _type_: _description_
        """

        h, _ = self.encoder(x)
        # Shape of h: (batch_size, dim_lstm)
        h = h[:, -1]
        # Shape of qyx: (batch_size, num_clusters)
        qyx = self.fc_y(h)
        # Shape of y: (batch_size, num_clusters)
        y = nn.functional.gumbel_softmax(qyx, hard=True)
        # Shape of zc: (batch_size, dim_z)
        zc = self.fc_cluster(y)

        # Shape of zb: (batch_size, dim_z)
        zb = self.fc_zb(h)
        # Convert zb to Standard distribution
        zb = (zb - self.zb_mean) / self.zb_std
        # Shape of z: (batch_size, dim_z)
        z = zc + zb

        seq_len = x.shape[1]
        # Shape of xc: (batch_size, seq_len, dim_x)
        xc = self.lstm_decode(zc, seq_len)
        # Shape of xp: (batch_size, seq_len, dim_x)
        xp = self.lstm_decode(z, seq_len)

        return y, zb, xc, xp
    

def compute_entropy(y):
    """
    Computes the entropy of a probability distribution tensor `y`.

    Args:
        y (torch.Tensor): Tensor of shape (batch_size, seq_len, num_clusters), 
                          representing probability distributions.

    Returns:
        torch.Tensor: Entropy tensor of shape (batch_size, seq_len).
    """
    # Ensure numerical stability by adding a small epsilon before log
    eps = 1e-9
    entropy = -torch.sum(y * torch.log(y + eps), dim=-1)
    return entropy

def vambc_loss(x, y, zb, xc, xp):
    loss_recon = nn.functional.mse_loss(xp, x)
    loss_kl = torch.sum(zb ** 2, dim=-1).mean()
    loss_ne = -compute_entropy(y).mean()
    loss_center = torch.sum((x - xc) ** 2, dim=-1).mean()
    loss = loss_recon + loss_kl + loss_ne + loss_center
    return loss