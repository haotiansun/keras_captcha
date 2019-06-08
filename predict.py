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

characters = string.digits + string.ascii_uppercase + string.ascii_lowercase
print(characters)

width, height, n_len, n_class = 140, 80, 4, len(characters)

model = load_model('2_cnn.h5')


def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y

def evaluate(model, batch_num = 20):
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
