import torch 
from torchmetrics import Accuracy 
from torchmetrics.classification import MulticlassAccuracy
from torch import nn, optim
import pandas as pd 
import numpy as np
from timeit import default_timer as timer
from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"   # DEVICE AGNOSTIC CODE 

#GETTING DATA  
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

class_names = train_data.classes
class_names
class_names_idx = train_data.class_to_idx
class_names_idx

# PREPARING DATA LOADERS WITCH SPECIFIED BATCH SIZE
BATCH_SIZE = 32 

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_data_loader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

# DEFINING TRAIN, TEST, EVALUATION STEPS 
def train_step(
    models: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn: Accuracy,
    device: device):
  train_loss, train_acc = 0, 0
  for batch, (X, y) in enumerate(data_loader):
    X, y = X.to(device), y.to(device)
    y_pred = models(X)
    loss = loss_fn(y_pred, y)
    acc = Accuracy(y_pred.argmax(dim=1), y)
    train_loss += loss
    train_acc += acc
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc*100}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn: Accuracy,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
      for X, y in data_loader:
          # Send data to GPU
          X, y = X.to(device), y.to(device)
            
          # 1. Forward pass
          test_pred = model(X)
            
          # 2. Calculate loss and accuracy
          test_loss += loss_fn(test_pred, y)
          test_acc += Accuracy(test_pred.argmax(dim=1), y) # Go from logits -> pred labels
            
        
      # Adjust metrics and print out
      test_loss /= len(data_loader)
      test_acc /= len(data_loader)
      print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc*100}%\n")

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn: MulticlassAccuracy, 
               device: torch.device = device):
    """Evaluates a given model on a given dataset.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += Accuracy(y_pred.argmax(dim=1), y)
        
        # Scale loss and acc
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

# SETTING OUR ACCURACY FUNC 

Accuracy = MulticlassAccuracy(num_classes=len(class_names)).to(device)

#FIRST BASIC MODEL WITH LINEAR LAYERS 
class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = number of features in a data sample (784 pixels)
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)

model_0 = FashionMNISTModelV0(input_shape=784, # one for every pixel (28x28)
    hidden_units=10, # how many units in the hidden layer
    output_shape=len(class_names) # one for every class
)
model_0

# SETTING LOSS AND OPTIMIZER FUNCS
loss_fn0 = nn.CrossEntropyLoss()
optimizer0 = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# TRAINING AND TESTING WHILE TAKING TIME TO SEE THE SPEED OF THE MODEL + MONITORING 

torch.manual_seed(42)
train_time_start_on_cpu = timer()
class_names = test_data.classes

# Set the number of epochs (we'll keep this small for faster training times)
epochs = 3

# Create training and testing loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(model_0, train_dataloader, loss_fn, optimizer, Accuracy, device)
    test_step(test_data_loader, model_0, loss_fn, Accuracy, device)

train_time_end_on_cpu = timer()
total_train_time_model_0 = train_time_end_on_cpu - train_time_start_on_cpu
print(f"Total training time: {total_train_time_model_0:.3f} seconds")


# PREPARING A CNN MODEL 

class FashionMNISTModelCNN(nn.Module):
  """Model architecture that replicates the TinyVGG
  model from CNN explainer site"""
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*7*7, out_features=output_shape)
    )

  def forward(self, x):
    x=self.conv_block_1(x)
    x=self.conv_block_2(x)
    x=self.classifier(x)
    
    return x


# INITIATING THE MODEL
torch.manual_seed(42)
model = FashionMNISTModelCNN(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)
model


#SETTING LOSS AND OPTIMIZER FUNCS 

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr = 0.1)


# TRAINING AND TESTING + MONITORING 

torch.manual_seed(42)
torch.cuda.manual_seed(42)

train_time_start_model = timer()

epochs = 3

for epoch in tqdm(range(epochs)):
  print(f"Epoch: {epoch}\n---------")
  train_step(model, train_dataloader, loss_fn, optimizer, Accuracy, device)
  test_step(test_data_loader, model, loss_fn, Accuracy, device)

train_time_end_model = timer()
total_train_time_model = train_time_end_model - train_time_start_model
print(f"Total training time: {total_train_time_model:.3f} seconds")


# CREATING RESULT DICTIONARIES
model_results = eval_model(
    model=model,
    data_loader=test_data_loader,
    loss_fn=loss_fn,
    accuracy_fn=Accuracy,
    device=device
)
model_results

model_0_results = eval_model(
    model=model_0,
    data_loader=test_data_loader,
    loss_fn=loss_fn0,
    accuracy_fn=Accuracy,
    device=device
)

model_0_results

#CREATING A DATAFRAME TO EASILY COMPARE MODELS' PERFORMANCE
import pandas as pd 

compare_results = pd.DataFrame([model_results, model_0_results])

compare_results

#+ADDITION+ IF NEEDED TIME CAN BE ADDED TOO.
compare_results["training_time"] = [total_train_time_model,
                                    total_train_time_model_0]

compare_results
