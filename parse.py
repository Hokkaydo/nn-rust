import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('output.csv', header=None, names=['epoch', 'loss'])
delta = 0.01
min_index = data['loss'].idxmin()
# data = data[data['loss'] <= delta + data['loss'][min_index]]

plt.plot(data['epoch'], data['loss'], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid()
plt.savefig('loss_plot.png')
