#load the data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return x_train, y_train, x_test, y_test

#data_path = 'mnist.npz'
#x_train, y_train, x_test, y_test = load_data(data_path)

#define activation functions
def relu(x):
    return x

def drelu(x):
    return 1. * (x > 0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)
def dersigmoid(x):
    return x * (1-x)


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis, keepdims=True))
    return e_x / e_x.sum(axis, keepdims=True)

#define loss metric
def cross_entropy(y_pred,y_true):
    return np.mean(-np.sum(y_true*np.log(y_pred), axis=1))

#one hot encoder
def one_hot(y_true,num_class):
    array = np.zeros([y_true.shape[0],num_class], dtype=y_true.dtype)
    for i, value in enumerate(y_true):
        array[i,value] = 1
    return array

def update_params(grad,params,lr):
    w1 = params[0] - lr * grad[0]
    b1 = params[1] - lr * grad[1]
    w2 = params[2] - lr * grad[2]
    b2 = params[3] - lr * grad[3]
    return w1,b1,w2,b2

def accuracy(model,x,y):
    counter = 0
    for i, (image, label) in enumerate(zip(x, y)):
        y_pred = int(model.predict(image))
        y_true = int(np.argmax(label))
        if y_pred == y_true:
            counter +=1
    return counter/len(x)

def preprocess(x_train, y_train, x_test, y_test):
    y_train = one_hot(y_train,10)
    y_test = one_hot(y_test,10)
    x_train = x_train/255
    x_test = x_test/255
    return x_train, y_train, x_test, y_test

def der_w(dz,a):
    a1_reshaped = a[...,np.newaxis]
    dz2_reshaped = dz[...,np.newaxis]
    dz2_reshaped = np.transpose(dz2_reshaped, (0, 2, 1))
    dw2 = a1_reshaped @ dz2_reshaped
    dw2 = np.mean(dw2, axis=0)
    return dw2

def train(model, x_train, y_train, x_test, y_test, batch_size, lr, epochs):
    for epoch in range(epochs):
        for i in range(0,len(x_train),batch_size):
            Xbatch = x_train[i:i+batch_size]
            Ybatch = y_train[i:i+batch_size]
            grad = model.grad(Xbatch,Ybatch)
            w1,b1,w2,b2 = update_params(grad,model.params(),lr)
            model.set_params(w1,b1,w2,b2)
            val_loss = model.forward(x_test)[3]
            val_loss = cross_entropy(val_loss, y_test)
            loss = grad[4]

    return model, loss, val_loss

def load_image(data_path):
    image = Image.open(data_path)
    image = ImageOps.grayscale(image)
    image = np.asarray(image, dtype=np.float32)
    image = 1 - (image/255)
    return image

