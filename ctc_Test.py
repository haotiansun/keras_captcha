from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import string
characters = string.digits + string.ascii_uppercase + string.ascii_lowercase
print(characters)

width, height, n_len, n_class = 140, 80, 4, len(characters)+1

from keras import backend as K
from keras.layers import merge
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    

def merge(inputs, mode, concat_axis=-1):
    return concatenate(inputs, concat_axis)




from keras.models import *
from keras.layers import *
rnn_size = 128

input_tensor = Input((width, height, 3))
x = input_tensor
for i in range(3):
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)

x = Dense(32, activation='relu')(x)

#gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')
#gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(x)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')

#gru1_merged = merge([gru_1, gru_1b], mode='sum')
x = Bidirectional(gru_1, merge_mode = 'sum')(x)
x = Bidirectional(gru_1b, merge_mode = 'sum')(x)

#gru1_merged = Bidirectional([gru_1, gru_1b], merge_mode='sum')

#gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')

#gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')

x = Bidirectional(gru_2, merge_mode = 'concat')(x)
x = Bidirectional(gru_2b, merge_mode = 'concat')(x)
#x = merge([gru_2, gru_2b], mode='concat')
#x = Bidirectional([gru_2, gru_2b], merge_mode='sum')(x)

x = Dropout(0.25)(x)
x = Dense(n_class, init='he_normal', activation='softmax')(x)
base_model = Model(input=input_tensor, output=x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')

from IPython.display import SVG, Image
#from keras.utils.visualize_util import plot, model_to_dot
#from keras.utils import plot_model
#plot(model, show_shapes=True)
#Image('model.png')
# SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

def gen(batch_size=128):
    X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
    y = np.zeros((batch_size, n_len), dtype=np.uint8)
    while True:
        generator = ImageCaptcha(width=width, height=height)
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = np.array(generator.generate_image(random_str)).transpose(1, 0, 2)
            y[i] = [characters.find(x) for x in random_str]
        yield [X, y, np.ones(batch_size)*int(conv_shape[1]-2), np.ones(batch_size)*n_len], np.ones(batch_size)
        
def evaluate(model, batch_num=8):
    batch_acc = 0
    generator = gen(128)
    for i in range(batch_num):
        [X_test, y_test, _, _], _  = next(generator)
        y_pred = base_model.predict(X_test)
        shape = y_pred[:,2:,:].shape
        out = K.get_value(K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0])[:, :4]
        if out.shape[1] == 4:
            batch_acc += ((y_test == out).sum(axis=1) == 4).mean()
    print(batch_acc / batch_num)
    return batch_acc / batch_num
    
from keras.callbacks import *

class Evaluate(Callback):
    def __init__(self):
        self.accs = []
    def on_train_begin(self, logs={}):
        self.losses = []
        self.train_acc = []
        self.val_losses = []
        self.c1_losses = []
        self.c2_losses = []
        self.c3_losses = []
        self.c4_losses = []
        #self.val_c1_losses = []
        #self.val_c2_losses = []
        #self.val_c3_losses = []
        #self.val_c4_losses = []
        self.c1_acc = []
        self.c2_acc = []
        self.c3_acc = []
        self.c4_acc = []
        #self.val_c1_acc = []
        #self.val_c2_acc = []
        #self.val_c3_acc = []
        #self.val_c4_acc = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        #self.c1_losses.append(logs.get('c1_loss'))
        #self.c2_losses.append(logs.get('c2_loss'))
        #self.c3_losses.append(logs.get('c3_loss'))
        #self.c4_losses.append(logs.get('c4_loss'))
        self.train_acc.append(logs.get('acc'))
        #self.c1_acc.append(logs.get('c1_acc'))
        #self.c2_acc.append(logs.get('c2_acc'))
        #self.c3_acc.append(logs.get('c3_acc'))
        #self.c4_acc.append(logs.get('c4_acc'))
    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(base_model)*100
        self.accs.append(acc)
        print ('acc: %f%%'%acc)
csv_logger = CSVLogger('18_log_0606_1024.csv', append=True, separator=';')
history = Evaluate()
model.fit_generator(gen(128), samples_per_epoch=80, nb_epoch=60,
                    callbacks=[EarlyStopping(patience=10), history, csv_logger],
                    validation_data=gen(), nb_val_samples=1024)
data = pd.DataFrame({"loss" : history.losses, "val_loss" : history.val_losses, "train_acc" : history.train_acc})
header = ["loss", "val_loss", "train_acc"]
data2 = pd.DataFrame({"accs": history.accs})
header2 = ["accs"]
data.to_csv('18_20190606_loss_acc.csv', encoding = 'utf-8', columns = header)                    
data2.to_csv('18_epoch_acc.csv', encoding = 'utf-8', columns = header2)   
evaluate(base_model)
model.save('18_model.h5')
