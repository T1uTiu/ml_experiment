import torch
from torch import nn
import numpy as np
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

def read_data(addr):
    data = np.loadtxt(addr, delimiter=',')

    n = data.shape[0]
    
    ###### You may modify this section to change the model
    X = torch.tensor(data[:,0:8]/2,dtype=torch.float32)
    ###### You may modify this section to change the model
    
    Y = None
    if "train" in addr:
        Y = torch.tensor(np.expand_dims(data[:, 8], axis=1),dtype=torch.float32)
    
    return (X,Y,n)

X,Y,n = read_data("train.csv")

class Net(nn.Module):
    def __init__(self,d_in):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(d_in, 10),
            nn.Sigmoid(),
            nn.Linear(10,10),
            nn.Sigmoid(),
            nn.Linear(10,10),
            nn.Sigmoid(),
            nn.Linear(10,1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):

        X = self.sigmoid(self.sequence(X))

        return X

model = Net(8)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

batch_size = 25
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_epochs = 100

for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


