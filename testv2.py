import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from workshopattacks import *

# define regression model
class Net(nn.Module):
    def __init__(self, dimension):
        super(Net, self).__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, 100),
                                    nn.Hardswish(),
                                    nn.Linear(100, 100),
                                    nn.Hardswish(),
                                    nn.Linear(100, 100),
                                    nn.Hardswish(),
                                    nn.Linear(100, 100),
                                    nn.Hardswish(),
                                    nn.Linear(100, 100),
                                    nn.Hardswish(),
                                    nn.Linear(100, 1)
                                    )
    
    def forward(self, x):
        x = self.layers(x)
        return x

def train(model, criterion, optimiser, epochs, x_train, y_train, x_test, y_test):
    for i in range(epochs):
        optimiser.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimiser.step()
        print(i, loss, criterion(model(x_test), y_test))

if __name__ == "__main__":
    # load data
    data = pd.read_csv("wineQualityReds.csv")
    X = data.drop(columns=["Unnamed: 0", "quality"]).to_numpy()
    y = data["quality"].to_numpy()
    n_samples, n_features = X.shape
    
    # train/test split
    rng = np.random.RandomState(0)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=rng)
    
    x_train = torch.Tensor(x_train)
    x_test = torch.Tensor(x_test)
    #x_train = F.normalize(x_train, dim=1)
    #x_test = F.normalize(x_test, dim=1)
    y_train = torch.Tensor(y_train).reshape(-1, 1)
    y_test = torch.Tensor(y_test).reshape(-1, 1)
    
    # train network
    net = Net(n_features)
    criterion = nn.MSELoss()
    optimiser = torch.optim.SGD(net.parameters(), lr=0.0005)
    train(net, criterion, optimiser, 500, x_train, y_train, x_test, y_test)
    
    # show some test predictions
    print(y_test[0:5])
    print(net(x_test[0:5]))
    
    print("Attack")
    e = 0.3317 * attack11(net, torch.abs(torch.cov(x_train.T)), x_test[0])
    #e = 0.3317 * attack11(net, torch.abs(torch.diag(torch.diagonal(torch.cov(x_train.T)))), x_test[0])
    #e = 0.3317 * attack11(net, torch.eye(11), x_test[0])
    print(torch.linalg.norm(e, ord=2))
    print(torch.linalg.norm(net(x_test[0]) - net(x_test[0] + e), ord=2))
    #print(y_test[0])
    #print(net(x_test[0] + e))
    
    print("FGSM")
    ex = FGSM(net, nn.MSELoss(), x_test[0], y_test[0], 0.1)
    print(torch.linalg.norm(x_test[0] - ex, ord=2))
    print(torch.linalg.norm(net(x_test[0]) - net(ex), ord=2))
    
    print("IFGSM")
    ex = IFGSM(net, nn.MSELoss(), x_test[0], y_test[0], 0.1)
    print(torch.linalg.norm(x_test[0] - ex, ord=2))
    print(torch.linalg.norm(net(x_test[0]) - net(ex), ord=2))