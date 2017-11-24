import env
import matplotlib.pyplot as plt
import numpy as np

e = env.WormFoodEnv((10, 10))
e.plot()
plt.show()

# for i in range(10):
#     print e.step(i * np.ones(20))
#     e.render()
#
# while True:
#     plt.pause(0.05)
