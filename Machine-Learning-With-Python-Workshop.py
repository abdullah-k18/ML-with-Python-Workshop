import numpy as np
import matplotlib.pyplot as plt

scores = np.array([5, 10, 50, 18, 74, 50, 9, 66])
balls_faced = np.array([18, 15, 58, 14, 92, 65, 16, 63])

x = balls_faced
y = scores

w1 = 1
w0 = 1
line = w1 * x + w0
plt.scatter(x, y)
plt.plot(x, line, 'g')
plt.show()

def loss (w1, w0, x, y):
  err = np.sum((w1 * x + w0) - y) ** 2
  return err

def grad(w1, w0, x, y):
  grad_w1 = np.sum(2 * ((w1 * x + w0) - y) * x)
  grad_w0 = np.sum(2 * ((w1 * x + w0) - y) * 1)
  return grad_w1, grad_w0

def update(w1, w0, grad_w1, grad_w0, gamma):
  w1 = w1 - gamma * grad_w1
  w0 = w0 - gamma * grad_w0
  return w1, w0


for epoch in range(1000):
    grad1, grad0 = grad(w1, w0, x, y)
    gamma = 0.00001
    w1, w0 = update(w1, w0, grad1, grad0, gamma)

err = loss(w1, w0, x, y)
print(err)
line2 = w1 * x + w0
plt.scatter(x, y)
plt.plot(x, line2, 'r')
plt.show()