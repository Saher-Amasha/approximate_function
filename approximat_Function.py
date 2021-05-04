import numpy as np
import matplotlib.pyplot as plt


def fourier_approx(f, max_iterations=25, speed=0.5, precision=1000):
    # f the function to approximate
    # max_iterations how many iterations the program will do
    # the time between each iteration
    # the number of point in each line space

    # define precision points in [-pi,pi]
    x = np.linspace(-np.pi, np.pi, precision)
    # defines an array with size precision where all values are 0
    y = np.zeros(precision)
    # difference between two points
    dx = x[1] - x[0]

    a0 = (1 / np.pi) * np.trapz(f(x), x, dx)
    y += (a0 / 2)

    for n in range(1, max_iterations):
        y += np.trapz(f(x) * np.cos(n * x), x, dx) * np.cos(n * x) / np.pi
        y += np.trapz(f(x) * np.sin(n * x), x, dx) * np.sin(n * x) / np.pi

        # plot the current approximated function in red and the original function in blue
        plt.plot(x, f(x), 'b')
        plt.plot(x, y, 'r')
        title = "n=" + str(n)
        plt.title(title)
        plt.show(block=False)
        plt.pause(speed)
        plt.clf()
    return


# function to approximate
def func(x): return x ** 2 + x**3 - 36*x


if __name__ == '__main__':
    fourier_approx(func, 75, 0.2, 1000)
