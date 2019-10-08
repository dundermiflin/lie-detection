import numpy as np
import pandas as pd
import json
import os
import io
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Embedding, Bidirectional, LSTM, concatenate
from keras.utils.vis_utils import plot_model
from keras import regularizers
from utils import preprocess_from_df, tokenizer_from_json, save_confusion_matrix, clean_text

seq_length = 256
embedding_size = 200
num_meta_feat = 5
hidden_neurons = 64
vocabulary_size = 20000
relevant_columns = [3, 9, 10, 11, 12, 13, 2]
columns_to_normalise = [9, 10, 11, 12, 13]
label_column = 2
sentence_column = 3
unique_labels = ['pants-fire', 'false',  'half-true', 'barely-true','mostly-true', 'true']
tokenizer = None

def fetch_tokenizer(sentences):
    global tokenizer
    if tokenizer is not None:
        return tokenizer
    if os.path.isfile('models/tokenizer.json'):
        with open('models/tokenizer.json') as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
    else:
        tokenizer = Tokenizer(num_words= vocabulary_size)
        tokenizer.fit_on_texts(sentences)
    
        tokenizer_json = tokenizer.to_json()
        with io.open('models/tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    return tokenizer

def sent_to_seq(sentences):
    sequences = fetch_tokenizer(sentences).texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=seq_length)

    return padded_sequences

def prep_data(filepath, tokenizer):
    data = pd.read_csv(filepath, sep = '\t', header = None)
    data[sentence_column] = data[sentence_column].apply(lambda x: clean_text(x))
    data = data[relevant_columns].dropna()

    X, y = preprocess_from_df(data, unique_labels, normalise_columns = columns_to_normalise, label_column = label_column, encode = True, split = False)
    sentences = list(X[:, -1])
    X_meta = X[:,:-1]

    padded_sequences = sent_to_seq(sentences)

    return X_meta, padded_sequences, y
if __name__ == "__main__":  

    X_train_meta, X_train_sequences, y_train = prep_data('data/train2.tsv',tokenizer)
    X_val_meta, X_val_sequences, y_val = prep_data('data/val2.tsv',tokenizer)
    X_test_meta, X_test_sequences, y_test = prep_data('data/test2.tsv',tokenizer)

    
    model = None
    if not os.path.isfile('models/nn_multi.json'):
        nlp_input = Input(shape=(seq_length,), name='nlp_input')
        meta_input = Input(shape=(5,), name='meta_input')
        emb = Embedding(input_dim = vocabulary_size,output_dim = embedding_size)(nlp_input)
        nlp_out = Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.01)))(emb)
        x = concatenate([nlp_out, meta_input])
        x = Dense(hidden_neurons, activation='relu')(x)
        x = Dense(len(unique_labels), activation='softmax')(x)
        model = Model(inputs=[nlp_input , meta_input], outputs=[x])
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        plot_model(model, to_file='figures/nn_multi_model_plot.png', show_shapes=True, show_layer_names=True)
        model.fit([X_train_sequences, X_train_meta], y_train, validation_data=([X_val_padded, X_val[:,:-1]], y_val), batch_size = 128, epochs=2)

        model_json = model.to_json()
        with open("models/nn_multi.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("models/nn_multi_weights.h5")
        print("Saved model to disk")
    else:
        # load json and create model
        json_file = open('models/nn_multi.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("models/nn_multi_weights.h5")
        print("Loaded model from disk")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        plot_model(model, to_file='figures/nn_multi_model_plot.png', show_shapes=True, show_layer_names=True)

    _, acc = model.evaluate([X_val_sequences, X_val_meta], y_val)
    print('Validation Accuracy: %.2f' % (acc))

    _, acc = model.evaluate([X_test_sequences, X_test_meta], y_test)
    print('Test Accuracy: %.2f' % (acc))

    y_pred = np.argmax(model.predict([X_test_sequences, X_test_meta]), axis = 1)
    y_true = np.argmax(y_test, axis = 1)

    save_confusion_matrix(y_pred, y_true, unique_labels,'multi')