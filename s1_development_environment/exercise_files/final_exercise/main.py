import click
import torch
from torch import optim
from torch import nn
import numpy as np
from model import MyAwesomeModel

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    # model
    model = MyAwesomeModel()
    model.train()
    # data
    train_set, _ = mnist("../../../data/corruptmnist", batch_size=128)
    # optimizer 
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # loss function 
    criterion = nn.NLLLoss()
    # parameters 
    num_epochs = 50
    save_freq = 10
    # run epochs 
    losses = []
    epoch_losses = []
    for epoch_i in range(num_epochs):
        print(f"Training epoch {epoch_i+1} of {num_epochs}")
        epoch_loss = 0
        for (inp, lbl) in train_set:
            # print(inp.shape)
            # print(lbl.shape)
            # clear gradients 
            optimizer.zero_grad() 
            # forward pass 
            outputs = model(inp)
            # print(outputs.shape)
            # calc loss 
            loss = criterion(outputs, lbl)
            epoch_loss += loss.item()
            # update
            losses.append(loss.item())
            # get gradients 
            loss.backward()
            # update model params 
            optimizer.step()
        
        # append mean epoch loss
        epoch_losses.append(epoch_loss/len(train_set))
        
        if (epoch_i+1)%save_freq == 0:
            torch.save({
                "epoch" : epoch_i, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_losses[-1],
                }, f"model_{epoch_i}")
        
    print(losses)
    print(epoch_losses)


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint)["model_state_dict"])
    # model = torch.load(model_checkpoint)
    _, test_set = mnist("../../../data/corruptmnist", batch_size=1)
    res = []
    with torch.no_grad():
        for (inp, lbl) in test_set:
            out = model(inp)
            pred = torch.argmax(out, dim=1)
            res.append((pred == lbl).numpy())
        res = np.mean(res)
        print("Accuracy :", res)
    
        
    
    


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
