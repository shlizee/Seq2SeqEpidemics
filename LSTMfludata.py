import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import fludata

exec(compile(open("fludata.py", "rb").read(), "fludata.py", 'exec'))
#exec(compile(open("SIRfludata.py", "rb").read(), "SIRfludata.py", 'exec'))

scaler = MinMaxScaler(feature_range=(-1, 1))
maxval = seasons[0:-1].max()
minval = seasons[0:-1].min()
scaleddata = scaler.fit_transform(seasons.reshape(-1, 1))
full_data = torch.FloatTensor(scaleddata).view(-1)
train_data = full_data[:-52] #removing last season from training data

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_window = 52
train_inout_seq = create_inout_sequences(train_data, train_window)

print(train_inout_seq[0])

import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
model = LSTM()
loss_function = nn.L1Loss()#MSELoss()#L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 40
losses=[]
for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        losses.append(single_loss)
        single_loss.backward()
        optimizer.step()
    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

losses=[float(i) for i in losses]
lossperepoch  =[]
sep=int(len(losses)/epochs)
c=0
for i in range(0,len(losses),sep):
    lossperepoch.append(sum(losses[c*sep:c*sep+sep]))
    c=c+1
plt.plot(range(0,epochs),np.log(lossperepoch))
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error Loss')
plt.title('Loss Over Time')
plt.legend(['Training Loss'])
plt.show()
print("Epoch with minimum loss: ",np.argmin(lossperepoch))

fut_pred = 52

test_inputs = train_data.tolist()
predictions=[]
model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        pred=model(seq).item()
        test_inputs.append(pred)
        predictions.append(pred)
actual_predictions = np.array(predictions)

unscaled = scaler.inverse_transform(actual_predictions.reshape(-1, 1))

fig=plt.figure(figsize=(15, 10), dpi= 80, facecolor='w', edgecolor='k')
y_pos = np.arange(len(weeks))
plt.xticks(y_pos, weeks)
plt.plot(seasons[-1], label='real data', linewidth=3, color='black')
plt.plot(unscaled, label='LSTM predictions',linewidth = 3, color='orange')
plt.legend(prop={'size': 20})
plt.yticks(fontsize=14)
plt.title("LSTM Predictions",fontsize=30)
plt.xlabel("Weeks",fontsize=30)
plt.ylabel("Number of New Cases",fontsize=30)
plt.show()

#fig=plt.figure(figsize=(15, 10), dpi= 80, facecolor='w', edgecolor='k')
#y_pos = np.arange(len(weeks))
#plt.xticks(y_pos, weeks)
#plt.plot(seasons[-1], label='real data',linewidth =3, color='black')
#plt.plot(unscaled, label='LSTM predictions',linewidth =3,color='orange')
#plt.plot(I, label="Previous Season fit SIR", linestyle='-',linewidth =3, color='blue')
#plt.plot(I2, label="Smart 10wks SIR", linestyle='-.', linewidth =3, color='aqua')
#plt.plot(I3, label="Naive 10wks SIR", linestyle='--',linewidth =3, color='navy')
#plt.legend(prop={'size': 20})
#plt.yticks(fontsize=14)
#plt.title("Comparing LSTM and SIR Predictions",fontsize=30)
#plt.xlabel("Weeks",fontsize=30)
#plt.ylabel("Number of New Cases",fontsize=30)
#plt.show()