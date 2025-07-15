import numpy as np
import matplotlib.pyplot as plt

def hankel_mat(data):
    n = int(len(data)/2)
    mat = np.zeros((n,n))
    for i in range(n):
        mat[:,i] = data[i:i+n]
    return mat

# constants
ENERGY_THRESH = 0.99

# load the output data
part_data = np.load("../data/processed/CL_cold.npy")
dh_data = part_data[:,:,-1]

# center the data
# dh_data = dh_data - np.nanmean(dh_data)

# setup plot
fig,ax = plt.subplots()
for i in range(part_data.shape[0]):
    # print(i)
    hankel = hankel_mat(dh_data[i,:])
    # print(hankel)

    # compute the SVD
    try:
        _,S,_ = np.linalg.svd(hankel)

        # Compute energy thresholds
        S2 = np.square(S)
        total_energy = np.sum(S2)
        S2_norm = S2/total_energy
        energy_count = 0

        energies = []

        for i,energy in enumerate(S2_norm):
            energy_count += energy
            energies.append(energy_count)

        # fig,ax = plt.subplots()
        ax.plot(energies)
    except np.linalg.LinAlgError:
        pass

ax.set_title("Singular Value Energy")
ax.set_ylabel("Percent Energy")
ax.set_xlabel("Number of Singular Values")
ax.set_ylim(0.95,1.01)
ax.plot([7,7],[0,1.5], 'r--')
ax.plot([0,23],[ENERGY_THRESH, ENERGY_THRESH], 'r--')
plt.show()

