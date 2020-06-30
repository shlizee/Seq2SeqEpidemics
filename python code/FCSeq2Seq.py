import Seq2Seqfludata
exec(compile(open("Seq2Seqfludata.py", "rb").read(), "Seq2Seqfludata.py", 'exec'))

import Seq2SeqCovid19_45days
exec(compile(open("Seq2SeqCovid19_45days.py", "rb").read(), "Seq2SeqCovid19_45days.py", 'exec'))

[e_data,t_data]=create_encoder_target_pairs(test_data_as_array) # using three states since we want few-shots
batch_size = 2**11
epochs = 100

encoder_input_data = e_data
encoder_input_data, encode_series_mean = transform_series_encode(encoder_input_data)

decoder_target_data = t_data                   
decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean)

decoder_input_data = np.zeros(decoder_target_data.shape)
decoder_input_data[:,1:,0] = decoder_target_data[:,:-1,0]
decoder_input_data[:,0,0] = encoder_input_data[:,-1,0]

history = model_flu.fit([encoder_input_data, decoder_input_data], decoder_target_data, epochs = 100,verbose=0);
plt.plot(history.history['loss'])

plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error Loss')
plt.title('Loss Over Time')
plt.legend(['Train'])
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

[e_data,t_data]=create_encoder_target_pairs(train_data_as_array[5])
[preds_rolling, target, rolling_mean]=get_rolling_average(e_data, t_data, 7)
plt.plot(e_data[0], color='black')
plt.plot(range(len(e_data[0]),len(e_data[0])+len(preds_rolling)),preds_rolling,color='red',linestyle='--', label='Predictions')
plt.plot(range(len(e_data[0]),len(e_data[0])+len(target)),target,color='green', label='Data')
plt.plot(range(len(e_data[0]),len(e_data[0])+len(target)),rolling_mean,color='m', label='7-Day Moving Average Data')

plt.title(states_train[5])
plt.legend()
plt.show()

[e_data,t_data]=create_encoder_target_pairs(train_data_as_array[15])
[preds, target, rolling_mean]=get_rolling_average(e_data, t_data, 7)
plt.plot(e_data[0], color='black')
plt.plot(range(len(e_data[0]),len(e_data[0])+len(preds)),preds,color='red',linestyle='--', label='Predictions')
plt.plot(range(len(e_data[0]),len(e_data[0])+len(target)),target,color='green', label='Data')
plt.plot(range(len(e_data[0]),len(e_data[0])+len(target)),rolling_mean,color='m', label='7-Day Moving Average Data')

plt.title(states_train[15])
plt.legend()
plt.show()

[e_data,t_data]=create_encoder_target_pairs(train_data_as_array[20])
[preds, target, rolling_mean]=get_rolling_average(e_data, t_data, 7)
plt.plot(e_data[0], color='black')
plt.plot(range(len(e_data[0]),len(e_data[0])+len(preds)),preds,color='red',linestyle='--', label='Predictions')
plt.plot(range(len(e_data[0]),len(e_data[0])+len(target)),target,color='green', label='Data')
plt.plot(range(len(e_data[0]),len(e_data[0])+len(target)),rolling_mean,color='m', label='7-Day Moving Average Data')
plt.title(states_train[20])
plt.legend()
plt.show()