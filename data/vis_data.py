import numpy as np
import matplotlib.pyplot as plt

data = np.load('processed/CL_cold.npy')

while True:
    layer = int(input("Enter Layer: "))
    plt.plot(data[layer,:,0])
    plt.plot(data[layer,:,1])
    plt.show()
