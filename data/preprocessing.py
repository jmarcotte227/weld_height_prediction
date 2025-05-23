import numpy as np
from scipy.io import loadmat

# constants
NUM_LAYERS = 105
NUM_SEGS = 46
USED_PARTS = ['CL_cold', 'CL_hot']
DTYPES = [
          # 'vel_set',
          'vel_calc',
          # 'avg_temp',
          'max_temp',
          'voltage',
          'current',
          'feedrate',
          'energy',
          # 'dir',
          'dh',
          ]

# load the raw .mat file
raw_data = loadmat('raw/bent_tube_dataset.mat', simplify_cells=True)

# compile sequences of data for each layer
for part in USED_PARTS:
    part_data = raw_data[part]
    seq_list = np.zeros((NUM_LAYERS, NUM_SEGS, len(DTYPES)))
    for layer in range(NUM_LAYERS):
        layer_data = part_data[f'l{layer+1}']
        # TODO: May need to trim these a bit
        # seq_list[layer, :, 0] = layer
        for idx, key in enumerate(DTYPES):
            # recording all the parameters in the list above.
            # dh is the target and will be used in the las
            seq_list[layer, :,idx] = layer_data[key][1:-1]
    np.save(f'processed/{part}.npy', seq_list)
