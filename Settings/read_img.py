import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import sys, os

root_path = os.path.join(sys.path[0], '..')
sys.path.append(root_path)

img = mpimg.imread(os.path.join(root_path, 'Resources', 'Medium.png'))
print(img.shape)