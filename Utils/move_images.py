import shutil
import numpy as np

path1 = "/home/amitjans/Desktop/My_network/data/train_dataset/image"
path2 = "/home/amitjans/Desktop/My_network/data/val_dataset/image"

for i in range(186, 403):
    a = np.random.randint(0, 100, dtype=int)
    if a <= 20:
        for end in ('.png', '.tif', '.csv'):
            shutil.move(path1 + str(i) + end, path2 + str(i) + end)
