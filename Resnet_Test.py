from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import string
import keras
from keras import metrics
import cv2
import os
from keras import applications
characters = string.digits + string.ascii_uppercase + string.ascii_lowercase
print(characters)

width, height, n_len, n_class = 128, 128, 4, len(characters)


from keras.utils.np_utils import to_categorical

#def gen(dir, batch_size=32, file_list=[]):
def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    #files = os.listdir(dir)
    #if len(file_list) == 10240:
     #   file_list.clear()
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            #pilImage = generator.generate_image(random_str)
            #pilImage = cv2.cvtColor(np.asarray(pilImage), cv2.COLOR_RGB2BGR)
            #pilImage = pilImage[:,:,None]
            #pilImage = pilImage.astype('uint8')
            #pilImage = cv2.cvtColor(pilImage, cv2.COLOR_BGR2GRAY)
            #pilImage = pilImage[:,:,None]
            #print (pilImage)
            #X[i] = pilImage
            #X[i] = cv2.cvtColor(pilImage, cv2.COLOR_BGR2GRAY)
            #path = random.choice(files)
            #imagePixel = cv2.imread(dir + '/' + path, 1)
            #filename = path[:4]
            #X[i] = imagePixel
            #if filename in file_list:
             #   i = i - 1
              #  continue
            #else:
             #   file_list.append(filename)
            #print (filename)
            for j, ch in enumerate(random_str):
            #for j, ch in enumerate(filename):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y

def test_gen(test_dir, batch_size=32, test_file_list=[]):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    #generator = ImageCaptcha(width=width, height=height)
    files = os.listdir(test_dir)
    if len(test_file_list) == 1024:
        test_file_list.clear()
    while True:
        for i in range(batch_size):
            #random_str = ''.join([random.choice(characters) for j in range(4)])
            #X[i] = generator.generate_image(random_str)
            path = random.choice(files)
            imagePixel = cv2.imread(test_dir + '/' + path, 1)
            filename = path[:4]
            X[i] = imagePixel
            if filename in test_file_list:
                i = i - 1
                continue
            else:
                test_file_list.append(filename)
            #print (filename)
            #for j, ch in enumerate(random_str):
            for j, ch in enumerate(filename):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

 
#X, y = next(gen(1))
#plt.imshow(X[0])
#plt.title(decode(y))

#plt.show()


from keras.models import *
from keras.layers import *
from keras.callbacks import CSVLogger

base_model = applications.resnet50.ResNet50(weights = None, include_top=False, input_shape = (height, width, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
#x = Flatten()(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
model = Model(input=base_model.input, output=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
csv_logger = CSVLogger('17_log_0606_1024.csv', append=True, separator=';')
dir = './images_4_digits'
test_dir = './test_images_4_digits'
def evaluate(model, batch_num = 1024):
    batch_acc = 0
    generator = gen()
    for i in tqdm(range(batch_num)):
        #X, y = generator.next()
        X, y = next(generator)
        y_pred = model.predict(X)
        batch_acc += np.mean(list(map(np.array_equal, np.argmax(y, axis=2).T, np.argmax(y_pred, axis=2).T)))
    print (batch_acc / batch_num)
    return batch_acc / batch_num

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        #self.val_losses = []
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
        #self.val_losses.append(logs.get('val_loss'))
        self.c1_losses.append(logs.get('c1_loss'))
        self.c2_losses.append(logs.get('c2_loss'))
        self.c3_losses.append(logs.get('c3_loss'))
        self.c4_losses.append(logs.get('c4_loss'))

        self.c1_acc.append(logs.get('c1_acc'))
        self.c2_acc.append(logs.get('c2_acc'))
        self.c3_acc.append(logs.get('c3_acc'))
        self.c4_acc.append(logs.get('c4_acc'))

dir = './images_4_digits'
test_dir = './test_images_4_digits'

history = LossHistory()

file_list = []
test_file_list = []
model.fit_generator(gen(), steps_per_epoch=320, epochs=60, callbacks=[history, csv_logger], validation_data=gen(), nb_val_samples=1024)

final_accuracy = evaluate(model)

data = pd.DataFrame({"loss" : history.losses, "c1_loss" : history.c1_losses, "c2_loss" : history.c2_losses, "c3_loss" : history.c3_losses, "c4_loss" : history.c4_losses, "c1_acc" : history.c1_acc, "c2_acc": history.c2_acc, "c3_acc" : history.c3_acc, "c4_acc" : history.c4_acc, "accuracy": final_accuracy})
header = ["loss", "c1_loss", "c2_loss", "c3_loss", "c4_loss", "c1_acc", "c2_acc", "c3_acc", "c4_acc", "accuracy"]
data.to_csv('17_20190606_loss_acc.csv', encoding = 'utf-8', columns = header)

#final_accuracy = evaluate(model)

model.save('17_cnn.h5')

