import pandas as pd
import numpy as np
import re
import pickle
import os
import sys
import keras
import argparse
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras import regularizers
from datetime import datetime
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D, Bidirectional
from keras.layers import MaxPool1D, SpatialDropout1D, LSTM, GlobalMaxPooling1D, concatenate, GlobalAveragePooling1D
from keras.models import Model, Sequential, model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


path = "/home/titho/Downloads/"
dir_embedding = os.path.join(path,"dir_embedding")
path_model = os.path.abspath(os.path.join(path, 'coba'))


class Sentiment_Classifier_BiLSTM(object):
    def __init__(self, predict=False, evaluate=False):
        self.drop = 0.30        
        self.batch_size = 20             
        self.epochs = 15
        self.max_sequence = 50
        self.embedding_dim = 300
        self.max_features = 9000
        self.filename_model = 'sentiment_model.json'
        self.MODEL_PATH = os.path.join(path_model, 'sentiment_emotion')
        
        self.filename_weight = 'sentiment_weight.h5'
        self.LABEL_INDEX = {"positive": 2, "negative": 0, "neutral": 1}
        self.INDEX_LABEL = dict(((v, k) for (k, v) in self.LABEL_INDEX.items()))
        self.filename_tokenizer = 'tokenizer_words.pickle'
        if predict==True:
            try:
                self.model, self.tokenizer = self.load_model()
            except Exception:
                raise ValueError('model was not found!')
        if evaluate==True:
            self.evaluate_model()
                
    def clean_text(self, text):
        tweet = re.sub(r"http\S+", "", text)
        tweet = re.sub(r"@\S+","", tweet)
        tweet = re.sub(r"#\S+","", tweet)
        tweet = re.sub('[^a-zA-Z]', ' ', tweet)
        tweet = tweet.lower()
        tweet = tweet.split()
        tweet = [chunk for chunk in tweet if len(chunk)>3]
        ps = PorterStemmer()
        tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
        tweet = ' '.join(tweet)
            
        return tweet
    
    def sequences_and_padding(self, tokenizer, x):
        sequences = tokenizer.texts_to_sequences(x)
        return pad_sequences(sequences, maxlen=self.max_sequence, padding='pre')
    
    def tokenize_text(self, X_train):
        tokenizer  = Tokenizer(num_words = self.max_features)
        tokenizer.fit_on_texts(X_train)
        word_index = tokenizer.word_index
        with open(os.path.join(self.MODEL_PATH, 'tokenizer_words.pickle'), 'wb') as o:
            pickle.dump(tokenizer, o, protocol=pickle.HIGHEST_PROTOCOL)
        return tokenizer, word_index
    
    def split_corpus(self, dataset):
        cleaned = []
        for m in range (len(dataset)):
            tweet = self.clean_text(dataset['text'][m])
            dataset['text'][m] = self.clean_text(dataset['text'][m])
            if tweet!='':
                cleaned.append(tweet)
        dataset = dataset[dataset.text!='']
        dataset = dataset.reset_index(drop=True)
        dataset = dataset['target']
        dataset = dataset.to_frame()
        dataset.insert(0, "text", cleaned, True)
        
        X_val = dataset.sample(frac=0.1, random_state=200)
        X_train = dataset.drop(X_val.index)
        X_test = X_train.sample(frac=0.1, random_state=200)
        X_train = X_train.drop(X_test.index)
        
        y_train = X_train['target'].apply(lambda x: self.LABEL_INDEX[x])
        y_train = y_train.tolist()

        y_val = X_val['target'].apply(lambda x: self.LABEL_INDEX[x])
        y_val = y_val.tolist()

        y_test = X_test['target'].apply(lambda x: self.LABEL_INDEX[x])
        y_test = y_test.tolist()
        
        X_train = X_train['text'].tolist()
        X_val = X_val['text'].tolist()
        X_test = X_test['text'].tolist()
        
        y_train = to_categorical(np.asarray(y_train))
        y_test = to_categorical(np.asarray(y_test))
        y_val = to_categorical(np.asarray(y_val))
        
        return X_train, y_train, X_test, y_test, X_val, y_val

    def get_vectors(self,word_index):
        vocabs = open(os.path.join(dir_embedding, 'glove.6B.300d.txt'), encoding="utf8")
        n_vocabs = len(word_index) + 1
        word_vectors = {}
        for line in vocabs:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_vectors[word] = coefs
        word_embedding_weights = np.zeros((n_vocabs, self.embedding_dim))
        for word, idx in word_index.items():
            try:
                word_embedding_weights[idx, :] = word_vectors[word]
            except:
                word_embedding_weights[idx] = np.random.normal(0, np.sqrt(0.25), self.embedding_dim)
        del word_vectors
        return word_embedding_weights
    
    def get_model(self, word_index, word_embedding_weights):
        words_input = Input(shape=(self.max_sequence,), name='word_input')
        words_emb = Embedding(input_dim=len(word_index) + 1,
                       output_dim=self.embedding_dim,
                       weights=[word_embedding_weights],
                       input_length=self.max_sequence,
                       name='embed_word')(words_input)
        x = Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True,
                kernel_regularizer=regularizers.l2(0.01)))(words_emb)
        max_pool = GlobalMaxPooling1D()(x)
        max_avg = GlobalAveragePooling1D()(x)
        concatenated = concatenate([max_pool, max_avg])
        addition = Dense(300, activation = 'relu')(concatenated)
        x = Dropout(0.25)(addition)
        main_output = Dense(len(self.INDEX_LABEL), activation='softmax')(x)
        model = Model(words_input, main_output)
        return model
    
    def train(self,dataset):
        X_train, y_train, X_test, y_test, X_val, y_val = self.split_corpus(dataset)
        tokenizer, word_index = self.tokenize_text(X_train)
        X_train = self.sequences_and_padding(tokenizer, X_train)
        X_val = self.sequences_and_padding(tokenizer, X_val)
        X_test = self.sequences_and_padding(tokenizer, X_test)
        word_embedding_weights = self.get_vectors(word_index)
        model = self.get_model(word_index, word_embedding_weights)
        
        early_stopping = EarlyStopping(monitor='val_accuracy',
                                               patience=3,
                                               verbose=1,
                                               mode='max')
        
        optim = RMSprop(lr=1e-4, rho = 0.9)
        model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
        
        model.summary()
        
        print('Train the model...')
        model.fit(X_train, y_train, 
                  batch_size=self.batch_size,
                  epochs=self.epochs, callbacks=[early_stopping],
                  validation_data=(X_val, y_val))
        self.save_model(model)
        print('Test the model...')
        loss, acc = model.evaluate(X_test, y_test, batch_size=self.batch_size)
        print('Test loss:', loss)
        print('Test accuracy:', acc)
        
    def evaluate_model(self):
        model, tokenizer = self.load_model()
        eval_path = os.path.abspath(os.path.join(self.MODEL_PATH, 'eval_dir'))
        try:
            os.mkdir(eval_path)
            print("Folder Eval Created") 
        except FileExistsError:
            print("Folder has been exist")
        datenow = datetime.now()
        corpus = pd.read_json('twitter_airline.json')
        _, _, X_test, y_test, _, _ = self.split_corpus(corpus)
        X_test = self.sequences_and_padding(tokenizer, X_test)
        y_pred = model.predict(X_test)
        y_test = np.argmax(y_test, axis = 1)
        y_pred = np.argmax(y_pred, axis = 1)
        
        cm = confusion_matrix(y_test, y_pred)
        sum = 0
        for i in range (len(cm)):
               sum = sum + cm[i][i]
        akurasi = sum/len(y_test)
        
        with open(os.path.join(eval_path, 'eval_bidirect_airline_dataset_' + datenow.strftime("%Y%m%d%M%S") + '.txt'), 'w') as _filetxt:
            _filetxt.write("{}\n".format('-' * 50))
            _filetxt.write("{}\n".format('TEST SET EVALUATION {}'.format(datenow.strftime("%Y%m%d"))))
            _filetxt.write("{}\n".format('-' * 50))
            print('TEST SET EVALUATION')
            print('Accuracy: {}\n'.format(akurasi))
            _filetxt.write("{}\n".format(cm))
            _filetxt.write("{}\n".format('Accuracy: {}\n'.format(akurasi)))
            print(classification_report(y_test, y_pred))
            _filetxt.write("{}\n".format(classification_report(y_test, y_pred)))
            _filetxt.close()
    
    def save_model(self, model):
        print('Serialize the model...')
        model_json = model.to_json()
        with open(os.path.join(self.MODEL_PATH, self.filename_model), 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(os.path.join(self.MODEL_PATH, self.filename_weight))
        
    def load_model(self):
        with open(os.path.join(self.MODEL_PATH, self.filename_model), 'r') as f:
            loaded_model_json = f.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(os.path.join(self.MODEL_PATH, self.filename_weight))
        with open(os.path.join(self.MODEL_PATH, self.filename_tokenizer), 'rb') as o:
            tokenizer = pickle.load(o)   
        
        return model, tokenizer
    
    def classify(self, sentence):
        cleaned_sentence = self.clean_text(sentence)
        cleaned_sentence = self.sequences_and_padding(self.tokenizer, [cleaned_sentence]) 
        result = np.argmax(self.model.predict(cleaned_sentence), axis=1)
        predicted = self.INDEX_LABEL[result[0]]
        print(predicted)
        return predicted
    
    def classify_batch(self, sentences):
        cleaned = []
        for n in range (len(sentences)):
            tweet = self.clean_text(sentences[n])
            cleaned.append(tweet)
        x_test = self.sequences_and_padding(self.tokenizer, cleaned) 
        result = self.model.predict_on_batch(x_test)
        result = np.apply_along_axis (np.argmax, 1, result)
        predicted = [self.INDEX_LABEL[r] for r in result]
        print(predicted)
        return predicted
    
if __name__ == '__main__':    
    ap = argparse.ArgumentParser()
    ap.add_argument('-mode', default='test')
    ap.add_argument('-dataset', default='twitter.json')
    ap.add_argument('-text', default=None, help='input the text')
    args = vars(ap.parse_args())

    mode = args['mode']
    dataset = args['dataset']
    text = args['text']
    
    if mode == 'train':
        corpus = pd.read_json(dataset)
        coba = Sentiment_Classifier_BiLSTM(predict=False, evaluate=False)
        coba.train(corpus)
    elif mode == 'eval':
        coba_1 = Sentiment_Classifier_BiLSTM(predict=False, evaluate=True)
    elif mode == 'batch_predict':
        coba_2 = Sentiment_Classifier_BiLSTM(predict=True)
        texts = ['I love Virginia Beach', 'I hate going to Airport']
        hasil_batch = coba_2.classify_batch(texts)
    else:
        if text is not None:
            coba_3 = Sentiment_Classifier_BiLSTM(predict=True)
            hasil = coba_3.classify(text)
        else:
            print('Please input the text')
    
