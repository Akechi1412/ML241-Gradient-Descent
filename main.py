import numpy as np
import matplotlib.pyplot as plt

def grad(x):
    return 6*x + 2 + 4*np.cos(x)

def cost(x):
    return 3*x**2 + 2*x + 4*np.sin(x)

def gradient_descent(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)

if __name__ == '__main__':
    (x1, it1) = gradient_descent(0.1, -5)
    (x2, it2) = gradient_descent(0.1, 5)
    (x3, it3) = gradient_descent(0.01, -1)

    print('x1 = %.6f, cost = %.6f, after %d' % (x1[-1], cost(x1[-1]), it1))
    print('x2 = %.6f, cost = %.6f, after %d' % (x2[-1], cost(x2[-1]), it2))
    print('x3 = %.6f, cost = %.6f, after %d' % (x3[-1], cost(x3[-1]), it3))