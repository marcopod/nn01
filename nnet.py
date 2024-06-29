import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_flattened = x_train.reshape(x_train.shape[0], -1)

def init_params(input_size, hidden_size, output_size):
    np.random.seed(69)
    w1 = np.random.rand(hidden_size, input_size) - 0.5
    b1 = np.ones((hidden_size, 1)) * 0.1
    w2 = np.random.rand(output_size, hidden_size) - 0.5
    b2 = np.ones((output_size, 1)) * 0.1
    print(f"input_size: {input_size}")
    print(f"hidden_size: {hidden_size}")
    print(f"output_size: {output_size}")

    return w1, b1, w2, b2

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
    return exp_z / np.clip(sum_exp_z, a_min=1e-10, a_max=None)  # Avoid division by zero

def forw_pass(x, w1, b1, w2, b2):
    z1 = np.dot(x, w1.T) + b1.T
    a1 = relu(z1)
    z2 = np.dot(a1, w2.T) + b2.T
    a2 = softmax(z2)

    print(f"x: {x.shape}")
    print(f"w1: {w1.shape}")
    print(f"b1: {b1.shape}")
    print(f"w2: {w2.shape}")
    print(f"b2: {b2.shape}")
    print(f"z1: {z1.shape}")
    print(f"a1: {a1.shape}")
    print(f"z2: {z2.shape}")
    print(f"a2: {a2.shape}")

    
    return z1, a1, z2, a2


def one_hot(labels, num_classes):
    one_hot_encoded = np.zeros((labels.size, num_classes))
    one_hot_encoded[np.arange(labels.size), labels] = 1
    return one_hot_encoded

y_train_encoded = one_hot(y_train, 10)
y_test_encoded = one_hot(y_test, 10)

def loss(Y, A2):
    print(f"Y:, {Y.shape}")
    print(f"A2:, {A2.shape}")
    m = Y.shape[0]  # Ensure this dimension matches the batch size
    epsilon = 1e-10  # A small number to avoid log(0)
    log_likelihood = -np.log(A2[np.arange(m), Y.argmax(axis=1)] + epsilon)  # Correct indexing
    return np.mean(log_likelihood)

w1, b1, w2, b2 = init_params(784, 10, 10)

# z1, a1, z2, a2 = forw_pass(x_train_flattened[0], w1, b1, w2, b2)
# cost = loss(y_train_encoded, a2)
print(x_train_flattened[0])
