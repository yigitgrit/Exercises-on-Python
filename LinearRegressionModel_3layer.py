import requests
from pathlib import Path
from helper_functions import plot_predictions

#Downloading helper functions

if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download...")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)


#Creating data

weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias

#Creating Training and testing splits

train_splits = int(0.8 * len(X_regression))
X_train_regression, y_train_regression = X_regression[:train_splits], y_regression[:train_splits]
X_test_regression, y_test_regression = X_regression[train_splits:], y_regression[train_splits:]

#Plot visiual of data
plot_predictions(train_data=X_train_regression,
                 train_labels=y_train_regression,
                 test_data=X_test_regression,
                 test_labels=y_test_regression)

#Creating Model
model_2 = nn.Sequential(nn.Linear(in_features=1, out_features=100),
                        nn.Linear(in_features=100, out_features=100),
                        nn.Linear(in_features=100, out_features=1))

loss = torch.nn.L1Loss()
optimizer_reg = torch.optim.SGD(model_2.parameters(), lr=0.0001)

torch.manual_seed(42)
epochs = 10000

for epoch in range(epochs):
  model_2.train()
  y_preds_reg = model_2(X_train_regression)
  loss_reg = loss(y_preds_reg, y_train_regression)
  optimizer_reg.zero_grad()
  loss_reg.backward()
  optimizer_reg.step()

  model_2.eval()
  with torch.inference_mode():
    test_preds_reg = model_2(X_test_regression)
    test_loss_reg = loss(test_preds_reg, y_test_regression)

  if epoch %100 == 0:
    print(f"Epoch: {epoch} | Loss: {loss_reg:.5f} | Test loss: {test_loss_reg:.5f}")

#Making predictions and plot visual of data

from helper_functions import plot_predictions
model_2.eval()
with torch.inference_mode():
  y_preds_reg = model_2(X_test_regression)
  plot_predictions(train_data=X_train_regression,
                   train_labels=y_train_regression,
                   test_data=X_test_regression,
                   test_labels=y_test_regression,
                   predictions=y_preds_reg)
