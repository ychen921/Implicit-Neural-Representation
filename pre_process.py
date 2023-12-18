import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

img_path = r'data/'

image = cv2.imread(img_path + 'bird.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height, width, _ = image.shape
re_size = 150

# resize image
image = cv2.resize(image, (int(re_size), int(re_size)), interpolation= cv2.INTER_LINEAR)
plt.imshow(image)
plt.axis('off')
plt.show()
plt.close()

# Save the resize image
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
os.chdir(img_path)
cv2.imwrite("image.png", image)