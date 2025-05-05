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

#Comparision by plotting 
compare_results.set_index("model_name")["model_loss"].plot(kind="barh")
plt.xlabel("loss")
plt.ylabel("model")

#MAKING PREDICTIONS 
def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device=device):
  pred_probs = []
  model.to(device)
  model.eval()
  with torch.inference_mode():
    for sample in data:
      sample = torch.unsqueeze(sample, dim=0).to(device)

      pred_logits = model(sample)
      pred_prob = torch.softmax(pred_logits.squeeze(), dim=0)
      pred_probs.append(pred_prob.cpu())

  return torch.stack(pred_probs)


#VISUALIZING THE PREDICTIONS WITH SOME RANDOM DATA FROM OUR ORIGINAL DATA (NOT LOADED ONE TO TEST IT) 
import random
RANDOM_SEED = 42 
test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data), k=9):
  test_samples.append(sample)
  test_labels.append(label)

plt.imshow(test_samples[0].squeeze(), cmap="gray")
plt.title(class_names[test_labels[0]])

#PLOTTING PREDICTIONS AND VISUALIZING THEM IN A 9,9 FIGURE

plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3 

for i, sample in enumerate(test_samples):
  plt.subplot(nrows, ncols, i+1)
  plt.imshow(sample.squeeze(), cmap="gray")
  pred_label = class_names[pred_labels[i]]
  truth_label = class_names[test_labels[i]]
  title_text = f"Pred: {pred_label} | Truth: {truth_label}"
  if pred_label == truth_label:
    plt.title(title_text, fontsize=10, c="b")  #LABELS WILL BE BLUE IF MATCHES BECAUSE IM A COLORBLIND :)
  else:
    plt.title(title_text, fontsize=10, c="r")  #LABELS ARE RED WHEN MISSMATCH
  plt.axis(False)


# MAKING WHOLE PREDICTIONS WITH THE DATA 
y_preds = []
model.eval()
with torch.inference_mode():
  for X, y in tqdm(test_data_loader, desc="Making predictions..."):
    X, y = X.to(device), y.to(device)
    y_logits = model(X)
    y_pred_probs = torch.softmax(y_logits.squeeze(), dim=0).argmax(dim=1)
    y_preds.append(y_pred_probs.cpu())

y_pred_tensor = torch.cat(y_preds)
y_pred_tensor[:10], len(y_pred_tensor)

# IMPORTING CONFUSION MATRIX TO FURTHER EVALUATE AND SEE THE POSSIBLE ERROR LABELS
try:
  import torchmetrics, mlxtend
  print(f"mlxtend version: {mlxtend.__version__}")
  assert int(mlxtend.__version__.split(".")[1]) >= 19, "mlxtend version should be 0.19.0 or higher" 
except:
  !pip install torchmetrics -U mlxtend
  import torchmetrics, mlxtend
  print(f"mlxtend version: {mlxtend.__version__}")

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

confmat = ConfusionMatrix(task="Multiclass", num_classes=len(class_names))
confmat_tensor = confmat(
    preds=y_pred_tensor,
    target=test_data.targets
    )

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10, 7)
    )


# LAST BUT NOT LEAST, SAVING AND LOADING MODEL FOR FURTHER USAGE 
from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "CNN_FASHIONMNIST_MODEL"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to :{MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

# LOADING 
torch.manual_seed(42)

loaded_model = FashionMNISTModelCNN(input_shape=1, hidden_units=10, output_shape=len(class_names))
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model.to(device)

# COMPARING TWO MODEL IF LOADED WITHOUT ANY ERRORS
torch.manual_seed(42)

loaded_model_res = eval_model(
    model=loaded_model,
    data_loader=test_data_loader,
    loss_fn=loss_fn,
    accuracy_fn=Accuracy,
    device=device
)

loaded_model_res
model_results

torch.isclose(
    torch.tensor(model_results["model_loss"]),
    torch.tensor(loaded_model_res["model_loss"]),
    atol=1e-02 # ITS NOT NECESSARY, IF THE PRINT OF .isclose IS TURNING FALSE YOU CAN CHECK THE TOLERANCE LEVEL WITH atol=1e-02 (means check 0.12 only) atol=1e-08(means check 0.12345678)
    )
