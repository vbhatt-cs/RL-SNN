import pickle
import matplotlib.pyplot as plt
import numpy as np

d = []
for i in range(1):
    with open('logs/s' + str(i), 'rb') as f:
        e = pickle.load(f)
    # plt.plot(e.d_history)
    d.append(np.array(e.d_history))

plt.title('Distance to food')
plt.plot(sum(d) / len(d))
plt.xlabel('Time (in ms)')
plt.ylabel('Distance to food')
e.plot()
plt.show()
