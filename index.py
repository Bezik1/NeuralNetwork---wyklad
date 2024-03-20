import numpy as np
import matplotlib.pyplot as plt

def tanh_grad(x):
    return 1 - np.tanh(x)**2

def mse(y, y_pred):
    return (y - y_pred)**2

def mse_grad(y, y_pred):
    return -2 * (y - y_pred)

class NeuralNetwork():
    def __init__(self, epochs, learning_rate):
        # Weights
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.w3 = np.random.randn()

        # Biases
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0

        # Hyperparameters
        self.epochs = epochs
        self.learning_rate = learning_rate
    
    def forward(self, x):
        h1 = np.tanh(self.w1 * x + self.b1)
        h2 = np.tanh(self.w2 * h1 + self.b2)
        o = np.tanh(self.w3 * h2 + self.b3)

        return o
    
    def backpropagation(self, x, y, h1, h2, o):
        mse_w1_grad = mse_grad(y, o) * tanh_grad(self.w3 * h2 + self.b3) * self.w3 * tanh_grad(self.w2 * h1 + self.b2) * self.w2 * tanh_grad(self.w1 * x + self.b1) * x
        mse_w2_grad = mse_grad(y, o) * tanh_grad(self.w3 * h2 + self.b3) * self.w3 * tanh_grad(self.w2 * h1 + self.b2) * h1
        mse_w3_grad = mse_grad(y, o) * tanh_grad(self.w3 * h2 + self.b3) * h2

        mse_b1_grad = mse_grad(y, o) * tanh_grad(self.w3 * h2 + self.b3) * self.w3 * tanh_grad(self.w2 * h1 + self.b2) * self.w2 * tanh_grad(self.w1 * x + self.b1)
        mse_b2_grad = mse_grad(y, o) * tanh_grad(self.w3 * h2 + self.b3) * self.w3 * tanh_grad(self.w2 * h1 + self.b2)
        mse_b3_grad = mse_grad(y, o) * tanh_grad(self.w3 * h2 + self.b3)

        self.w1 -= self.learning_rate * mse_w1_grad
        self.w2 -= self.learning_rate * mse_w2_grad
        self.w3 -= self.learning_rate * mse_w3_grad

        self.b1 -= self.learning_rate * mse_b1_grad
        self.b2 -= self.learning_rate * mse_b2_grad
        self.b3 -= self.learning_rate * mse_b3_grad

    def train(self, X, Y):
        loss_history = []
        y_pred_list = []

        for epoch in range(self.epochs):
            epoch_loss_history = []
            for x, y in zip(X, Y):
                h1 = np.tanh(self.w1 * x + self.b1)
                h2 = np.tanh(self.w2 * h1 + self.b2)
                o = np.tanh(self.w3 * h2 + self.b3)

                loss = mse(y, o)
                epoch_loss_history.append(loss)
                self.backpropagation(x, y, h1, h2, o)
                
                if epoch == self.epochs-1:
                    y_pred_list.append(o)
            
            average_epoch_loss = sum(epoch_loss_history) / len(epoch_loss_history)
            loss_history.append(average_epoch_loss)
            print(f'Epoch: {epoch}, Average-Loss: {average_epoch_loss}')

        return loss_history, y_pred_list

X = np.linspace(-1, 1, 5000)
Y = np.array([x**2 for x in X])

NN = NeuralNetwork(100, 0.05)
loss_history, y_pred_list = NN.train(X, Y)

plt.plot(loss_history)
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(X[2000:3001], Y[2000:3001], label='Expected')
plt.plot(X[2000:3001], np.array(y_pred_list[2000:3001]).squeeze(), label='Predicted')
plt.title('Expected vs Predicted')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()