import numpy as np
import matplotlib.pyplot as plt

import env


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


rng = np.random.RandomState(0)
num_parts = 20
theta_max = 25.0
e = env.WormFoodEnv((-10, 10), num_parts=num_parts, theta_max=theta_max)

w = rng.uniform(-1 / theta_max, 1 / theta_max, (num_parts * 2, num_parts))
z = np.zeros(w.shape)
left_angle, right_angle = e.obs()
beta = 0.5
gamma = 0.1

T = 1000

print e.disToFood

for t in range(T):
    a = w.T.dot(np.hstack((left_angle, right_angle)))
    a = a.clip(-theta_max, theta_max)
    grad_a = np.tile(np.hstack((left_angle, right_angle)), (num_parts, 1)).T
    r, left_angle, right_angle, _ = e.step(a)
    z = beta * z + grad_a / a
    w += gamma * r * z

print e.disToFood
e.plot()
plt.figure()
plt.plot(e.d_history)

plt.show()
