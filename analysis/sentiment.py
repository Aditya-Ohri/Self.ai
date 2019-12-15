import fileinput
import numpy as np
from keras.models import Input, Model,load_model
from keras.layers import Bidirectional, LSTM, TimeDistributed, Dense, concatenate
from keras.callbacks import EarlyStopping
from typing import List, Dict, Tuple
from cortecx.construction.tools import *


class Parser:

    def __init__(self, fp: str=None, limit: int=10000, embeddings_matrix: Dict=None):
        self._file_path = fp
        self._limit = limit

        if embeddings_matrix is not None:
            self._embeddings_matrix = embeddings_matrix
        else:
            self._embeddings_matrix = {}

    def set_file_path(self, fp: str):
        if not isinstance(fp, str):
            raise TypeError('File path must be tpye str.')
        else:
            self._file_path = fp

    def set_limit(self, limit: int):
        if not isinstance(limit, int):
            raise TypeError('Limit must be type int')
        else:
            self._limit = limit

    def set_embeddings_matrix(self, embeddings_matrix: Dict):
        if not isinstance(self._embeddings_matrix, dict):
            raise TypeError('Embeddings matrix must be of type Dict')
        else:
            self._embeddings_matrix = embeddings_matrix

    def parse_embeddings(self):
        line_tracker = 0
        for line in fileinput.input([self._file_path]):
            if line_tracker > self._limit:
                fileinput.close()
                break
            else:
                splits = str(line).replace('\n', '').split(' ')
                wrd = splits[0]
                vector = [float(num) for num in splits[1:]]
                self._embeddings_matrix.update({wrd: vector})
                line_tracker += 1
                continue
        return self._embeddings_matrix

    @property
    def embeddings(self) -> Dict:
        return self._embeddings_matrix

    def word_vectorize(self, text: str) -> List:
        vectorized = []

        tokenizer = Tokenizer(text=text)
        tokens = tokenizer.tokenize()
        tokens = tokens.tokens

        for word in tokens:
            try:
                vectorized.append(self._embeddings_matrix[word])
            except KeyError:
                vectorized.append(np.zeros(300).tolist())
        return vectorized

    @staticmethod
    def char_vectorize(word: str, vector_matrix: Dict):
        try:
            return [vector_matrix[letter] for letter in word]
        except KeyError:
            return [np.zeros(300) for letter in word]

    def wipe(self):
        self._embeddings_matrix = {}


def load_data(fp: str, limit: int=1000) -> Tuple[List, List]:
    text = []
    targets = []
    for i, line in enumerate(fileinput.FileInput([fp])):
        if limit is not None:
            if i > limit:
                return text, targets
        if i > 0:
            line = line.split(',')
            line, target = clean(line[2].lower(), include_punctuation=False,
                                       include_numbers=False, filters='\n\t'), [float(line[1])]
            text.append(Tokenizer(line[1:]).tokenize().tokens)
            targets.append(target)
    fileinput.close()

    return text, targets


def encode_model_char_side(data: List, char_matrix: Dict, pad_len_word: int=10, pad_len_sentence: int=60):
    encoded = []
    for sentence in data:
        for i, word in enumerate(sentence):
            sentence[i] = padding(Parser.char_vectorize(word, char_matrix),
                                        pad_len=pad_len_word, pad_char=np.zeros(300).tolist())
        sentence = padding(sentence, pad_len=pad_len_sentence, pad_char=np.zeros(shape=(10, 300)).tolist())
        encoded.append(sentence)
    return encoded


def encode_model_word_side(data: List, vector_matrix: Dict):
    encoded = []
    for element in data:
        temp = []
        for word in element:
            try:
                temp.append(vector_matrix[word])
            except KeyError:
                temp.append(np.zeros(300).tolist())
        temp = padding(temp, pad_len=60, pad_char=np.zeros(300).tolist())
        encoded.append(temp)
    return encoded


class SentimentModel:

    def __init__(self):
        self.model = None

    def build_model(self):
        word_in_layer = Input(shape=(60, 300))
        char_in_layer = Input(shape=(60, 10, 300))

        char_branch = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False, recurrent_dropout=0.2)))(
            char_in_layer)

        tree = concatenate([word_in_layer, char_branch])
        tree = Bidirectional(LSTM(256, recurrent_dropout=0.2))(tree)
        output_layer = Dense(1, activation='sigmoid')(tree)

        model = Model([word_in_layer, char_in_layer], output_layer)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.model = model

    def train(self):
        train_data = load_data('../data/sentiment/train.csv', limit=100)

        parser = Parser()

        parser.set_limit(limit=100)
        parser.set_file_path('../data/word_vectors.txt')
        parser.parse_embeddings()

        print('Data Loaded')

        word_x_train = np.array(encode_model_word_side(train_data[0], parser.embeddings))
        parser.wipe()

        parser.set_file_path('../data/char_vectors.txt')
        parser.parse_embeddings()

        char_x_train = []

        parser.wipe()

        print(word_x_train.shape)
        print(char_x_train.shape)

        callbacks = EarlyStopping(monitor='val_accuracy', patience=5)
        self.model.fit([word_x_train, char_x_train], np.array(train_data[1]), epochs=1,
                       validation_split=0.3, callbacks=[callbacks])

    def save(self, fp: str):
        self.model.save(fp)


"""
sentiment_model = SentimentModel()
sentiment_model.build_model()
sentiment_model.train()
sentiment_model.save('sentiment_model.h5')
"""


class SentimentPredictor:

    def __init__(self):
        self._text = ''
        self._model = None
        self._word_matrix = {}
        self._char_matrix = {}

    def load_model(self, model_fp: str):
        self._model = load_model(model_fp)

    def load_char_embeddings(self, fp: str):
        parser = Parser()

        parser.set_file_path(fp=fp)
        parser.parse_embeddings()

        self._char_matrix = parser.embeddings
        parser.wipe()

    def load_word_embeddings(self):
        parser = Parser()

        parser.set_limit(limit=20000)
        parser.set_file_path('../data/word_vectors.txt')
        self._word_matrix = parser.parse_embeddings()

        parser.wipe()

    def __call__(self, *args, **kwargs):
        data = Tokenizer(args[0]).tokenize().tokens

        word_x_train = encode_model_word_side([data], self._word_matrix)
        char_x_train = encode_model_char_side([data], self._char_matrix)

        return self._model.predict([word_x_train, char_x_train])


predictor = SentimentPredictor()
predictor.load_char_embeddings('../data/char_vectors.txt')
predictor.load_word_embeddings()
predictor.load_model('../models/sentiment_model.h5')

print(predictor('This is a test sentence'))
