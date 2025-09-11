import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, d_in=10, d_hidden=32, d_out=2):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

#Testing code below (can be removed once done)

#Create an instance of the model you've defined
model = NeuralNetwork()

#Creating random data with batch size 2 (recall that there are 10 input neurons)
x = torch.randn(2, 10)
print("x: ", x)

#Calling the instance calls forward()
logits = model(x)
print("Logits: ", logits)
