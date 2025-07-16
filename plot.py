data = [
    [4, 0.000089597],
    [16, 0.000199717],
    [64, 0.000335518],
    [256, 0.00222258],
    [1024, 0.006403285],
    [4096, 0.031105541],
    [16384, 0.138928595],
    [65536, 0.634871993],
    [262144, 2.514987882],
    [1048576, 10.831404744],
    [4194304, 54.100831441],

]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.DataFrame(data, columns=['size', 'time'])

data['size'] = data['size'].astype(int)
data['time'] = data['time'].astype(float)
plt.plot(data['size'], data['time'], marker='o', label='Time vs Size')
plt.plot(data['size'], data['size'] * np.log(data['size']) * 1e-6, label='n log n', linestyle='--')
plt.plot(data['size'], data['size'] ** 2 * 1e-6, label='n²', linestyle='--')
plt.plot(data['size'], data['size'] * 1e-6, label='n', linestyle='--')

plt.xlabel('Size')
plt.ylabel('Time (seconds)')
plt.title('Time vs Size')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('time_vs_size.png')
