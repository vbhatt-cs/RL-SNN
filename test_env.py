import env
import matplotlib.pyplot as plt
import numpy as np

e = env.WormFoodEnv((2, 3))

for i in range(10):
    print e.step(i * np.ones(20))
    e.render()

while True:
    plt.pause(0.05)
