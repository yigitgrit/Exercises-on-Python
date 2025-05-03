import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from helper_functions import plot_decision_boundary
import requests
from pathlib import Path

# Downloading some files from net, in this case, helper_functions.py by MrDBourke.
# Its good to know how to use prepared unofficial libraries by downloading them.

if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download...")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

# Creating the data
n_samples = 1000

X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42
                    )
#Visualizing the data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)

#Converting data to Tensors

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Building a model with linear activation .ReLU
class CircleModelV2(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(in_features=2, out_features=10)
    self.layer_2 = nn.Linear(in_features=10, out_features=10)
    self.layer_3 = nn.Linear(in_features=10, out_features=1)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CircleModelV2()
model_3

# Picking a suitable loss and optimizer functions 

loss = nn.BCEWithLogitsLoss() # We need torch.sigmoid() before we actualy make some predictions because our loss function works with LOGITS, Careful about that, the logits need activation functions before we use.
optimizer = torch.optim.SGD(model_3.parameters(), lr = 0.1) #Picking stochastic gradient descent optimizer.

#Creating a training and testing phase for our model to run and evaluate the model

torch.manual_seed(42) # Manual seed to have close result in any computer.
epochs = 2000 # 2000 times the model will run and learn before making our predictions 

for epoch in range(epochs): #Training 
  model_3.train()
  y_logits = model_3(X_train).squeeze()
  y_pred = torch.round(torch.sigmoid(y_logits))
  Loss = loss(y_logits, y_train)
  acc = accuracy_fn(y_true=y_train, y_preds=y_pred)
  optimizer.zero_grad()
  Loss.backward()
  optimizer.step()

  model_3.eval() #Testing
  with torch.inference_mode():
    test_logits = model_3(X_test).squeeze()
    test_preds = torch.round(torch.sigmoid(test_logits))
    test_loss = loss(test_logits, y_test)
    test_acc = accuracy_fn(y_true=y_test, y_preds=test_preds)

  if epoch %100 == 0: # Printing what we have in more undarstandable way, .xf command is for 0.xxxxxx decimal points meaning (eg: .2f = 0.25, .5f=0.00035 etc.)
    print(f"Epoch: {epoch} | Loss: {Loss:.5f} | Acc: {acc:.2f}% | Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%")


# Making Predictions

model_3.eval()
with torch.inference_mode():
  y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()


#Visualizing our predictions with test and train data.

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_3, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test)
