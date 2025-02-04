from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import string
import keras
from keras import metrics
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import cv2
characters = string.digits + string.ascii_uppercase + string.ascii_lowercase
print(characters)

width, height, n_len, n_class = 140, 80, 4, len(characters)

model = load_model('14_cnn.h5')


#def gen(batch_size=32):
 #   X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
  #  y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
   # generator = ImageCaptcha(width=width, height=height)
    #while True:
     #   for i in range(batch_size):
      #      random_str = ''.join([random.choice(characters) for j in range(4)])
       #     X[i] = generator.generate_image(random_str)
        #    for j, ch in enumerate(random_str):
         #       y[j][i, :] = 0
          #      y[j][i, characters.find(ch)] = 1
        #yield X, y
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
            #X[i] = generator.generate_image(random_str)
            pilImage = generator.generate_image(random_str)
           # #pilImage = cv2.cvtColor(np.asarray(pilImage), cv2.COLOR_RGB2BGR)
            #pilImage = pilImage[:,:,None]
            #pilImage = pilImage.astype('uint8')
           # pilImage = cv2.cvtColor(pilImage, cv2.COLOR_BGR2GRAY)
          #  pilImage = pilImage[:,:,None]
            #print (pilImage)
            X[i] = pilImage
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

evaluate(model)
