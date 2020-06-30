import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.DataFrame(pd.read_csv('train.csv'))

US_data = data.loc[data['Country_Region'] == 'US']
US_data_cases = US_data.loc[US_data['Target'] == 'ConfirmedCases']
US_data_cases = US_data_cases.loc[US_data_cases['Population'] == 324141489]
fig=plt.figure(figsize=(10, 5), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(US_data_cases['TargetValue'].values/324141489*100, linewidth = 3)
plt.title("Daily Number of New Cases for US  by %population",fontsize=20)
plt.xlabel("Time in Days",fontsize=15)
plt.ylabel("Percent of Population Newly Infected",fontsize=15)
plt.show()

states_train =['Alaska', 'Arizona', 'Arkansas', 'California',
       'Colorado', 'Connecticut', 'Delaware', 'District of Columbia',
       'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois',
       'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine',
       'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
       'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
       'New Hampshire', 'New Jersey', 'New Mexico',
       'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
       'Pennsylvania', 'Rhode Island', 'South Carolina',
       'South Dakota', 'Tennessee',  'Utah', 'Vermont','Virginia', 'Washington', 'West Virginia',
       'Wisconsin', 'Wyoming']
states_test=['Alabama','New York','Texas']
df_train = pd.DataFrame(columns =['Date']+states_train)#columns = states
df_train["Date"] = US_data_cases["Date"].values

df_test = pd.DataFrame(columns =['Date']+states_test)#columns = states
df_test["Date"] = US_data_cases["Date"].values

train_data_as_array=[]
for state in states_train:
    state_data=US_data.loc[US_data['Province_State'] == state]
    state_pop = state_data['Population'].max()
    state_data_cases = state_data.loc[state_data['Target'] == 'ConfirmedCases']
    state_data_cases = state_data_cases.loc[state_data_cases['Population'] == state_pop]
    df_train[state]=state_data_cases['TargetValue'].values
    max_cases = df_train[state].max()
    min_cases = df_train[state].min()
    df_train[state]=(df_train[state]-min_cases)/(max_cases-min_cases)
    train_data_as_array.append(df_train[state].values)
train_data_as_array=np.vstack(train_data_as_array)

test_data_as_array=[]
for state in states_test:
    state_data=US_data.loc[US_data['Province_State'] == state]
    state_pop = state_data['Population'].max()
    state_data_cases = state_data.loc[state_data['Target'] == 'ConfirmedCases']
    state_data_cases = state_data_cases.loc[state_data_cases['Population'] == state_pop]
    df_test[state]=state_data_cases['TargetValue'].values
    max_cases = df_test[state].max()
    min_cases = df_test[state].min()
    df_test[state]=(df_test[state]-min_cases)/(max_cases-min_cases)
    test_data_as_array.append(df_test[state].values)
test_data_as_array=np.vstack(test_data_as_array)

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")


latent_dim = 50 # LSTM hidden units
dropout = .20 

encoder_inputs = Input(shape=(None, 1)) 
encoder = LSTM(latent_dim, dropout=dropout, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, 1)) 

decoder_lstm = LSTM(latent_dim, dropout=dropout, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)

decoder_dense = Dense(1) 
decoder_outputs = decoder_dense(decoder_outputs)

model_covid = Model([encoder_inputs, decoder_inputs], decoder_outputs)

def create_encoder_target_pairs(data):
    e_data = []
    t_data = []
    if len(data)!=119:
        for i in range(len(data)):
            e_data.append(data[i][0:-45])
            t_data.append(data[i][-45:])
    else:
        e_data.append(data[0:-45])
        t_data.append(data[-45:])
    return [np.array(e_data),np.array(t_data)]

def transform_series_encode(series_array):
    series_mean = series_array.mean(axis=1).reshape(-1,1) 
    series_array = series_array.reshape((series_array.shape[0],series_array.shape[1], 1))
    
    return series_array, series_mean

def transform_series_decode(series_array, encode_series_mean):
    series_array = series_array - np.zeros([len(encode_series_mean),1])#encode_series_mean
    series_array = series_array.reshape((series_array.shape[0],series_array.shape[1], 1))
    
    return series_array

[e_data,t_data]=create_encoder_target_pairs(train_data_as_array)

batch_size = 2**11
epochs = 100

encoder_input_data = e_data
encoder_input_data, encode_series_mean = transform_series_encode(encoder_input_data)

decoder_target_data = t_data
                                            
decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean)

decoder_input_data = np.zeros(decoder_target_data.shape)
decoder_input_data[:,1:,0] = decoder_target_data[:,:-1,0]
decoder_input_data[:,0,0] = encoder_input_data[:,-1,0]

model_covid.compile(Adam(), loss='mean_absolute_error')
history = model_covid.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_split=0.2,verbose=0);
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error Loss')
plt.title('Loss Over Time')
plt.legend(['Train','Valid'])
plt.show()

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs] + decoder_states)

def decode_sequence(input_seq,pred_steps):
    
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, 1))
    target_seq[0, 0, 0] = input_seq[0, -1, 0]

    decoded_seq = np.zeros((1,pred_steps,1))
    
    for i in range(pred_steps):
        
        output, h, c = decoder_model.predict([target_seq] + states_value)
        
        decoded_seq[0,i,0] = output[0,0,0]

        target_seq = np.zeros((1, 1, 1))
        target_seq[0, 0, 0] = output[0,0,0]

        states_value = [h, c]

    return decoded_seq


def get_cumulative_data(data):
    c_data = np.zeros((len(data),1))
    c_data[0]=data[0]
    for i in range(1,len(data)):
        c_data[i]=c_data[i-1]+data[i]
    return c_data

def get_cumulative_data_preds(edata, tdata):
    c_edata = get_cumulative_data(edata)
    c_data = np.zeros((len(tdata),1))
    c_data[0] = tdata[0]+c_edata[-1]
    for i in range(1,len(tdata)):
        c_data[i]=c_data[i-1]+tdata[i]
    return c_data
def get_rolling_average(edata, tdata, numsteps):
    c = 0
    pred_steps=numsteps
    rolling_preds=[edata[-1][-1]]
    for i in range(len(tdata[0])):
        encoder_input_data = np.array([np.append(edata[0],tdata[0][0:c])])
        encoder_input_data, encode_series_mean = transform_series_encode(encoder_input_data)
    
        decoder_target_data = [t_data[0][c:c+pred_steps]]
        decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean)
    
        rolling_preds= np.append(rolling_preds,np.mean(decode_sequence(encoder_input_data,pred_steps)))
        c=c+1
    target = np.append([edata[-1][-1]],tdata[0])
    rolling_data = pd.Series(target).rolling(window = numsteps)
    rolling_mean = rolling_data.mean()
    return rolling_preds, target, rolling_mean

def predict_and_plot(e_data, t_data, enc_tail_len=len(e_data[0])):
    pred_steps=len(t_data[0])
    encoder_input_data = e_data
    encoder_input_data, encode_series_mean = transform_series_encode(encoder_input_data)

    decoder_target_data = t_data
    decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean)
    
    encode_series = encoder_input_data
    pred_series = decode_sequence(encode_series,pred_steps)
    
    encode_series = encode_series.reshape(-1,1)
    pred_series = pred_series.reshape(-1,1)   
    
    target_series = decoder_target_data.reshape(-1,1) 
    target_series =np.concatenate([encode_series[-1:],target_series])
    
    encode_series_tail = encode_series[-enc_tail_len:]
    x_encode = encode_series_tail.shape[0]
    
    #to show continuity
    preds = np.zeros((pred_steps+1,1))
    preds[0]=encode_series_tail[-1]
    preds[1:]=pred_series
    
    
    plt.figure(figsize=(20,6))  
    plt.subplot(1, 2, 1)  
    plt.plot(range(x_encode,x_encode+pred_steps+1),target_series,color='green')
    plt.plot(range(x_encode,x_encode+pred_steps+1),preds,color='red',linestyle='--')
    plt.plot(range(1,x_encode+1),encode_series_tail, color='black')
    plt.legend(['Target Series','Predictions'])
    
    [preds_rolling, target, rolling_mean]=get_rolling_average(e_data, t_data, 7)
    plt.subplot(1, 2, 2) 
    plt.plot(range(x_encode,x_encode+pred_steps+1),target,color='green', label = 'Target Series')
    plt.plot(range(x_encode,x_encode+pred_steps+1),preds_rolling,color='red',linestyle='--',label = '7-Day Moving Average Predictions' )
    plt.plot(range(x_encode,x_encode+pred_steps+1),rolling_mean,color='m', label='7-Day Moving Average Data')
    plt.legend()
    plt.plot(range(1,x_encode+1),encode_series_tail, color='black')
    return pred_series

[e_data,t_data]=create_encoder_target_pairs(test_data_as_array[0])
preds= predict_and_plot(e_data,t_data)
plt.suptitle("Alabama")
plt.show()

[e_data,t_data]=create_encoder_target_pairs(test_data_as_array[1])
preds= predict_and_plot(e_data,t_data)
plt.suptitle("New York")
plt.show()

[e_data,t_data]=create_encoder_target_pairs(test_data_as_array[2])
preds= predict_and_plot(e_data,t_data)
plt.suptitle("Texas")
plt.show()