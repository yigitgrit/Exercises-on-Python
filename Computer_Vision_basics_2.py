import torch
from torch import nn, optim
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt 
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random





# Setup device agnostic code

device = "cuda" if torch.cuda.is_available() else "cpu"

# Creating Data from torchvision MNIST dataset 

train = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

# Preparing batches of data 

train_data_loader = DataLoader(dataset=train, batch_size=32, shuffle=True)
test_data_loader = DataLoader(dataset=test, batch_size=32, shuffle=False)

# Creating labels parameter as class_names(labels)

class_names = train.classes

# Print to be sure everything goes well 

print(f"Train data: {len(train_data_loader.dataset)}")
print(f"Test data: {len(test_data_loader.dataset)}")
print(len(class_names))



# Visualizing a data to see if our data is matching with what we actually want

sample = train_data_loader.dataset[18][0]
print(f"Sample shape: {sample.shape}")
import matplotlib.pyplot as plt

plt.imshow(sample.squeeze(), cmap="gray")
plt.axis("off")
plt.title(train_data_loader.dataset[18][1])
plt.show()




# Creating a model as TinyVGG

from torch import nn, optim
class MNISTModel(nn.Module):
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super().__init__()
    self.cnn_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.cnn_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*7*7,
                  out_features=output_shape)
    )
  
  def forward(self, x):
    x=self.cnn_block_1(x)
    x=self.cnn_block_2(x)
    x=self.classifier(x)
    return x
    
model_0 = MNISTModel(
    input_shape=1,
    hidden_units=10,
    output_shape=len(class_names)
).to(device)

# Checking our agnostic code and seeing info about our assigned model

device, model_0


# Testing our model with a dummy input tensor 

dummy = torch.rand(size=(1, 28, 28)).unsqueeze(dim=0).to(device)
model_0(dummy)

# Training and testing for N epochs ( in this case 5 epochs ) 

torch.manual_seed(42)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr = 0.1)

epochs = 5 

for epoch in tqdm(range(epochs)):
  train_loss_total = 0 
  for batch, (X, y) in enumerate(train_data_loader):
    model_0.train()
    X, y = X.to(device), y.to(device)
    y_logits = model_0(X)
    loss = loss_fn(y_logits, y)
    train_loss_total += loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  train_loss_total /= len(train_data_loader)

  test_loss_total = 0
  
  model_0.eval()
  with torch.inference_mode():
    for batch, (X, y) in enumerate(test_data_loader):
      X, y = X.to(device), y.to(device)
      test_logits = model_0(X)
      test_loss = loss_fn(test_logits, y)
      test_loss_total += test_loss

    test_loss_total /= len(test_data_loader)

  print(f"Epoch: {epoch} | Train Loss: {train_loss_total:.5f} | Test Loss: {test_loss_total:.5f}")


# Defining a prediction function 

def predictions(model:torch.nn.Module, data:list, device:torch.device=device):
  predictions = []
  model.eval()
  with torch.inference_mode():
    for sample in data:
      sample = torch.unsqueeze(sample, dim=0).to(device)
      pred_logits = model(sample)
      pred_probs = torch.softmax(pred_logits.squeeze(), dim=0)
      predictions.append(pred_probs.cpu())
  return torch.stack(predictions)

# Testing our predictions with randomized values 

test_samples = []
test_labels = []

for sample, label in random.sample(list(test), k=9):
  test_samples.append(sample)
  test_labels.append(label)


pred_probs = predictions(model=model_0, data=test_samples)
pred_classes = pred_probs.argmax(dim=1)
pred_classes[:2], pred_probs[:2]

# Checking the performance by visualizing with predictions and truths

plt.figure(figsize=(12, 12))
rows = 3
cols = 3 

for i, sample in enumerate(test_samples):
  plt.subplot(rows, cols, i+1)
  plt.imshow(sample.squeeze(), cmap="gray")
  pred_label = class_names[pred_classes[i]]
  truth_label = class_names[test_labels[i]]
  plt.axis(False)
  plt.title(f"Pred: {pred_label}, | Truth: {truth_label}")


# Make predictions with trained model
y_preds = []
model_0.eval()
with torch.inference_mode():
  for X, y in tqdm(test_data_loader, desc="Making predictions"):
    # Send data and targets to target device
    X, y = X.to(device), y.to(device)
    # Do the forward pass
    y_logit = model_0(X)
    # Turn predictions from logits -> prediction probabilities -> predictions labels
    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
    # Put predictions on CPU for evaluation
    y_preds.append(y_pred.cpu())
# Concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)


# Checking torchmetrics and mlxtend 
try:
    import torchmetrics, mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")
    assert int(mlxtend.__version__.split(".")[1]) >= 19, "mlxtend verison should be 0.19.0 or higher"
except:
    !pip install -q torchmetrics -U mlxtend # <- Note: If you're using Google Colab, this may require restarting the runtime
    import torchmetrics, mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")


# Creating a confusion matrix to deeply understand where our model is wrong/right

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# 2. Setup confusion matrix instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test.targets)

# 3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
    class_names=class_names, # turn the row and column labels into class names
    figsize=(10, 7)
);


# A CONFUSION MATRIX LOOK LIKE ( I like to use them often )
* https://canada1.discourse-cdn.com/flex029/uploads/roboflow1/original/2X/4/4b2d20c22a8e0f2aa785d083f111fa68da71d961.png  # NORMALIZED ( 0's are dropped)
* https://www.researchgate.net/publication/330817434/figure/fig4/AS:962461243023361@1606480050428/Confusion-matrix-of-the-three-CNN-models-a-normalized-confusion-matrix-of-one-layer-CNN.png

