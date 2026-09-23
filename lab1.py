# Lab 1 - Search Spaces, Benchmark Functions and Random Search
#
# What is a Search Space?
# In simple words: a search space is the set of every possible answer an
# algorithm is allowed to try. If you are looking for the lowest point in a
# landscape, the search space is the entire landscape - every (x, y) location
# you could stand on. A benchmark function is just a standard, well-known
# "landscape" that researchers use to test and compare optimization algorithms fairly.

import numpy as np
import matplotlib.pyplot as plt
import random


# ---------------------------------------------------------------
# Sphere function
# ---------------------------------------------------------------
def sphere(x, y):
    return x**2 + y**2


x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)

X, Y = np.meshgrid(x, y)
Z = sphere(X, Y)

plt.contourf(X, Y, Z, levels=30, cmap='viridis')
plt.colorbar(label='f(x, y)')
plt.title('Sphere Function Landscape')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# ---------------------------------------------------------------
# Rastrigin function
# ---------------------------------------------------------------
def rastrigin(x, y):
    A = 10
    return 2*A + (x**2 - A*np.cos(2*np.pi*x)) \
        + (y**2 - A*np.cos(2*np.pi*y))


x = np.linspace(-5.12, 5.12, 200)
y = np.linspace(-5.12, 5.12, 200)

X, Y = np.meshgrid(x, y)
Z = rastrigin(X, Y)

plt.contourf(X, Y, Z, levels=30, cmap='viridis')
plt.colorbar(label='f(x, y)')
plt.title('Rastrigin Function Landscape')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# ---------------------------------------------------------------
# Ackley function
# ---------------------------------------------------------------
def ackley(x, y):
    return (-20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2)))
            - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))
            + np.e + 20)


x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)

X, Y = np.meshgrid(x, y)
Z = ackley(X, Y)

plt.contourf(X, Y, Z, levels=30, cmap='viridis')
plt.colorbar(label='f(x, y)')
plt.title('Ackley Function Landscape')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# ---------------------------------------------------------------
# Random Search
# ---------------------------------------------------------------
def random_search(func, bounds, iterations=1000):
    best_point = None
    best_score = float('inf')

    for i in range(iterations):
        x = random.uniform(bounds[0], bounds[1])
        y = random.uniform(bounds[0], bounds[1])

        score = func(x, y)

        if score < best_score:
            best_score = score
            best_point = (x, y)

    return best_point, best_score


best_point, best_score = random_search(sphere, (-5, 5))
print('Sphere -> Best point:', best_point,
      'Best score:', best_score)

best_point, best_score = random_search(rastrigin, (-5.12, 5.12))
print('Rastrigin -> Best point:', best_point,
      'Best score:', best_score)

best_point, best_score = random_search(ackley, (-5, 5))
print('Ackley -> Best point:', best_point,
      'Best score:', best_score)


# ---------------------------------------------------------------
# Questions & Answers
# ---------------------------------------------------------------
# 1. Which function did Random Search solve best? Why do you think that
#    happened, based on its landscape shape?
#    Random Search solved the Sphere function best, as it got a score of
#    0.0403, which is very close to the minimum value of 0. The Sphere function
#    has a simple and smooth landscape with only one minimum point at (0,0),
#    so random points can easily get close to the best solution.
#
# 2. Which function did Random Search struggle with the most? What about its
#    landscape made it difficult?
#    Random Search struggled the most with the Rastrigin function, with a best
#    score of 0.4606. Its landscape has many small valleys and local minima,
#    which can make it difficult for a random search to find the exact global
#    minimum. However, in my run it still found a point quite close to (0,0).
#
# 3. If you increased iterations from 1,000 to 10,000, do you expect the
#    results to improve? For which function would the improvement matter
#    least, and why?
#    Yes, I expect the results to improve because the algorithm would get 10
#    times more chances to find a better point. The improvement would probably
#    matter least for the Sphere function, because it already achieved a very
#    good score of 0.0403 with only 1,000 iterations.
#
# 4. In your own words, what is the main weakness of Random Search as a strategy?
#    The main weakness of Random Search is that it has no strategy or
#    direction. It simply chooses random points and keeps the best one.
#    Because of this, it can waste many attempts and may miss the actual best
#    solution, especially in complicated search spaces with many valleys or
#    flat areas.
