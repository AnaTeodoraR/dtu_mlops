from torch import nn

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(784, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 10),
                                    nn.Softmax(dim=1))
        
    def forward(self, x):
        return self.layers(x)