""" Exercise about binary and multiclass classifications """ 
# importing all we need to run 
# you may need to install torchmetrics pip -q install torchmetrics

import torch
from sklearn.datasets import make_moons
import pandas as pd
import numpy as np 
from torchmetrics import Accuracy 
from torch import nn, optim
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

#Setting device agnostic code for priorizing cuda
device = "cuda" if torch.cuda.is_available() else "cpu"

#Setting random seed to have similar outputs for every user 
RANDOM_SEED = 42

#Creating data 
n_samples = 1000

X, y = make_moons(n_samples=n_samples, noise=0.05, random_state=RANDOM_SEED)

#Building a data frame to see better 
moons = pd.DataFrame({"X_feature_1" : X[:, 0],
                     "X_feature_2" : X[:, 1],
                     "labels" : y})

# run this to see the table we made: moons.head(10)

#Visualizing what are we dealing with and determining rather binary or multiclass etc. 
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("DATA")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)

#Preparing train test splits 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)

#Building the model and initiating it
inf = 2
hid = 16
outf = 1

class MoonModelV0(nn.Module):
  def __init__(self, input_features, hidden_units, output_features):
    super().__init__()
    self.layer_stack = nn.Sequential(
        nn.Linear(in_features=inf, out_features=hid),
        nn.ReLU(),
        nn.Linear(in_features=hid, out_features=hid),
        nn.ReLU(),
        nn.Linear(in_features=hid, out_features=outf)
    )
  def forward(self, x):
    return self.layer_stack(x)

moon_model = MoonModelV0(inf, hid, outf).to(device)
# if you want to see what your model does run: moon_model

#Picking loss and optimizer functions 
loss_fn = nn.BCEWithLogitsLoss() # Since the problem is binary 
optimizer = torch.optim.Adam(params=moon_model.parameters(), lr = 0.01) # Better than normal SGD in most cases

#Creating our accuracy function 
acc_fn = Accuracy(task="multiclass", num_classes=2).to(device)
acc_fn

# Preparing a training and testing loop 
torch.manual_seed(RANDOM_SEED)

epochs = 100 



for epoch in range(epochs):
  moon_model.train()
  y_moon_logits = moon_model(X_train)
  y_moon_preds = torch.round(torch.sigmoid(y_moon_logits)).squeeze()
  loss = loss_fn(y_moon_logits.squeeze(), y_train)
  acc = acc_fn(y_moon_preds, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  moon_model.eval()
  with torch.inference_mode():
    test_logits = moon_model(X_test)
    test_preds = torch.round(torch.sigmoid(test_logits)).squeeze()
    test_loss = loss_fn(test_logits.squeeze(), y_test)
    test_acc = acc_fn(test_preds, y_test)
    
  if epoch % 10 == 0:
    print(f"Epoch: {epoch} | Loss: {loss} | Acc: {acc} | Test Loss: {test_loss} | Test Acc: {test_acc}")


# To visualize we are creating a function called plot_decision_boundary ( which was installed from helper_functions earlier in this repo ) 
def plot_decision_boundary(model, X, y):
  
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Source - https://madewithml.com/courses/foundations/neural-networks/ 
    # (with modifications)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), 
                         np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # mutli-class
    else: 
        y_pred = torch.round(torch.sigmoid(y_logits)) # binary
    
    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Visualizing both Train and Test results 
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(moon_model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(moon_model, X_test, y_test)



### SECOND CASE 

# Creating a spiral shaped data and checking it by visualizing

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
plt.show()

#Creating our data tensors
X = torch.from_numpy(X).type(torch.float) # features as float32
y = torch.from_numpy(y).type(torch.LongTensor) # labels need to be of type long

# Creating train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

#Defining our accuracy function 
acc_fn = Accuracy(task="multiclass", num_classes=3).to(device)
acc_fn

#Creating the model and initializing it 
ins = 2
hids = 16
outs = 3 
class modelv1(nn.Module):
  def __init__(self, ins, hids, outs):
    super().__init__()
    self.linear_layer_stack = nn.Sequential(
        nn.Linear(in_features=ins, out_features=hids),
        nn.ReLU(),
        nn.Linear(in_features=hids, out_features=hids),
        nn.ReLU(),
        nn.Linear(in_features=hids, out_features=outs)
    )
  
  def forward(self, x):
    return self.linear_layer_stack(x)

model = modelv1(ins, hids, outs)
# To see what model is doing run: model

# Picking loss and optimizer functions 
loss_fn = nn.CrossEntropyLoss() # Since its now a multiclass we cannot use nn.BCEWithLogits we use Cross Entropy instead.
optimizer = torch.optim.Adam(params=model.parameters(), lr = 0.02) 

# Creating Training and Testing loop 
torch.manual_seed(RANDOM_SEED)
epochs = 100 

for epoch in range(epochs):
  model.train()
  y_logits = model(X_train)
  y_probs = torch.softmax(y_logits, dim=1)
  y_preds = y_probs.argmax(dim=1)
  loss = loss_fn(y_logits, y_train)
  acc = acc_fn(y_preds, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  model.eval()
  with torch.inference_mode():
    test_logits = model(X_test)
    test_probs = torch.softmax(test_logits, dim=1)
    test_preds = test_probs.argmax(dim=1)
    test_loss = loss_fn(test_logits, y_test)
    test_acc = acc_fn(test_preds, y_test)

  if epoch % 10 == 0:
    print(f"Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}% ")

# Visualizing and checking the results and performance of our model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)



