#Linear Regression Model + Visualizing the data using plotlib + Save/load model

import torch
import torch.optim
from torch import nn
import matplotlib.pyplot as plt
import pathlib as Path
import pickle



#Setting the hyper parameters
weight = 0.3
bias = 0.9
start = 0
end = 1
step = 0.01
lr = 0.01

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
train_length = int(0.8 * len(X))
x_train = X[:train_length]
y_train = y[:train_length]
x_test = X[train_length:]
y_test = y[train_length:]

def plot_predictions(train_data = x_train,
                     train_label = y_train,
                     test_data = x_test,
                     test_label = y_test,
                     predictions = None
                     ):
    plt.figure(figsize=(10,5))
    plt.scatter(train_data, train_label, c="b", s=6, label="Training Data")
    plt.scatter(test_data, test_label, c="r", s=6, label="Testing Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="y", label="Predictions")
    plt.legend(prop={"size": 15});
    plt.show()


# if you want to see what the curve should be with training data
# plot_predictions();

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(X)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
model = LinearRegressionModel()

# if you want to check current dict
# print(model.state_dict())

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
epochs = 300
epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    model.train()
    preds = model(x_train)
    loss = loss_fn(preds, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_preds = model(x_test)
        test_loss = loss_fn(test_preds, y_test)

    if epoch %20 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")


with torch.inference_mode():
    new_preds = model(x_test)
    plot_predictions(predictions=new_preds)
    plt.show()

with open("model_state_dict", "wb") as f: # wb stand for write mode rb stands for read mode
    pickle.dump(model.state_dict(), f)

with open("model_state_dict", "rb") as f:
    model_load_state_dict = pickle.load(f)

print(model.state_dict())
print(model_load_state_dict)

# SOURCE_DIR = "/opt/anaconda3/envs/Experiments/lib/python3.10/pathlib.py"
# MODEL_PATH = Path(SOURCE_DIR, "models")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)
#
# SAVE_NAME = "first model linear regression state dict"
# SAVE_PATH = MODEL_PATH / SAVE_NAME
#
#pathlib is not working for some reason so we use pickle and open commands
# torch.save(obj=model.state_dict(), f=SAVE_PATH)
