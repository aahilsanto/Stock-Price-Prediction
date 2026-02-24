# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

The objective of this work is to design and implement a deep learning model for predicting stock closing prices using historical time-series data. The dataset consists of training and testing stock price records, where only the “Close” price is considered for analysis. The data is normalized and converted into sequential input samples to capture temporal dependencies over a fixed time window. A Recurrent Neural Network (RNN) model is developed using PyTorch to learn patterns from past price movements. The model is trained using Mean Squared Error loss and optimized with the Adam optimizer. Finally, the trained model is evaluated on unseen test data, and the predicted prices are compared with actual prices to analyze forecasting performance.


## Design Steps

### Step 1:

Load the training and testing datasets and extract the closing price column for analysis. Normalize the data using MinMax scaling to ensure stable and faster convergence during training.

### Step 2:

Convert the normalized time-series data into sequential input-output pairs using a fixed window size. This helps the model learn temporal dependencies from past stock prices.

### Step 3:

Transform the generated sequences into PyTorch tensors and create a DataLoader for batch-wise training. This enables efficient data handling and model optimization.

### Step 4:

Define a Recurrent Neural Network model with multiple hidden layers and a fully connected output layer. Configure the loss function as Mean Squared Error and use the Adam optimizer for training.

### Step 5:

Train the model over multiple epochs and monitor the training loss to evaluate learning progress. Plot the loss curve to analyze convergence behavior.

### Step 6:

Use the trained model to predict stock prices on the test dataset. Perform inverse scaling and compare predicted values with actual prices to evaluate forecasting performance.



## Program
#### Name: Ahil Santo A
#### Register Number: 212224040018
Include your code here
```Python 
# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
    super(RNNModel,self).__init__()
    self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
    self.fc=nn.Linear(hidden_size,output_size)

  def forward(self,x):
    out,_=self.rnn(x)
    out=self.fc(out[:,-1,:])
    return out


model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)


# Train the Model
epochs=20
model.train()
train_loss=[]
for epoch in range(epochs):
  epoch_loss=0
  for x_batch,y_batch in train_loader:
    x_batch,y_batch=x_batch.to(device),y_batch.to(device)
    optimizer.zero_grad()
    outputs=model(x_batch)
    loss=criterion(outputs,y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss+=loss.item()
  train_loss.append(epoch_loss/len(train_loader))
  print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss[-1]:.4f}")
```

## Output

### True Stock Price, Predicted Stock Price vs time

<img width="1194" height="781" alt="image" src="https://github.com/user-attachments/assets/a78a202a-40b7-4be2-bd30-3f82d7cc16f3" />


### Predictions 

<img width="881" height="137" alt="image" src="https://github.com/user-attachments/assets/01e2e0b6-601f-4ec1-a6f3-4675154fdace" />

## Result

The RNN model was successfully trained to predict stock closing prices, and the predicted values closely followed the actual prices with reduced Mean Squared Error.
