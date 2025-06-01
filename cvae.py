import torch
import torch.nn as nn
import torch.nn.functional as F
class CVAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super(CVAE,self).__init__()
        self.num_classes= num_classes
        self.latent_dim=latent_dim
        self.label_emb= nn.Embedding(num_classes, 28*28)

        self.encoder=nn.Sequential(
            nn.Linear(28*28*2,400),
            nn.ReLU()
        )
        self.fc_mu =nn.Linear(400,latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 10, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid()
        )

    def encode(self, x, y):
            y_embed = self.label_emb(y)
            x_concat = torch.cat([x.view(-1, 28 * 28), y_embed], dim=1)
            h = self.encoder(x_concat)
            return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

    def decode(self, z, y):
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        z = torch.cat([z, y_onehot], dim=1)
        return self.decoder(z)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z, y)
        return x_reconst, mu, logvar
