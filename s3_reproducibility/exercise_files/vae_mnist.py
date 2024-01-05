"""Adapted from https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb.

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import os
import hydra 
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import Decoder, Encoder, Model
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

# Model Hyperparameters
# dataset_path = "~/datasets"
# cuda = True
# DEVICE = torch.device("cuda" if cuda else "cpu")
# batch_size = 100
# x_dim = 784
# hidden_dim = 400
# latent_dim = 20
# lr = 1e-3
# epochs = 20

@hydra.main(config_path="conf", config_name="config.yaml")
def train(cfg):
    # set up
    hyperparams = cfg.experiments
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(hyperparams.seed)
    log = logging.getLogger(__name__)
    
    # Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(hyperparams.data_path, transform=mnist_transform, train=True, download=True)
    test_dataset = MNIST(hyperparams.data_path, transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=hyperparams.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=hyperparams.batch_size, shuffle=False)

    encoder = Encoder(input_dim=hyperparams.x_dim, 
                      hidden_dim=hyperparams.hidden_dim, 
                      latent_dim=hyperparams.latent_dim)
    decoder = Decoder(latent_dim=hyperparams.latent_dim, 
                      hidden_dim=hyperparams.hidden_dim, 
                      output_dim=hyperparams.x_dim)

    model = Model(encoder=encoder, decoder=decoder).to(device)
    
    
    def loss_function(x, x_hat, mean, log_var):
        """Elbo loss function."""
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + kld


    optimizer = Adam(model.parameters(), lr=hyperparams.lr)
    
    log.info("Start training VAE...")
    model.train()
    for epoch in range(hyperparams.n_epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                log.info(batch_idx)
            x = x.view(hyperparams.batch_size, hyperparams.x_dim)
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
        log.info(f"Epoch {epoch+1} complete!,  Average Loss: {overall_loss / (batch_idx*hyperparams.batch_size)}")
    log.info("Finish!!")

    # save weights
    torch.save(model, f"{os.getcwd()}/trained_model.pt")

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            if batch_idx % 100 == 0:
                log.info(batch_idx)
            x = x.view(hyperparams.batch_size, hyperparams.x_dim)
            x = x.to(device)
            x_hat, _, _ = model(x)
            break

    save_image(x.view(hyperparams.batch_size, 1, 28, 28), "orig_data.png")
    save_image(x_hat.view(hyperparams.batch_size, 1, 28, 28), "reconstructions.png")

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(hyperparams.batch_size, hyperparams.latent_dim).to(device)
        generated_images = decoder(noise)

    save_image(generated_images.view(hyperparams.batch_size, 1, 28, 28), "generated_sample.png")

if __name__ == "__main__":
    train()