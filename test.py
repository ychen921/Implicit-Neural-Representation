import torch
from numpy.ma.core import size
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.io import read_image
import matplotlib.patches as patches

from data_reader import SingleImageDataset
from model import FFN

img_path = r'data/'
model_path = img_path + 'save_model.pt'
img = read_image(img_path+'image.png')
gt_image = torch.zeros(img.shape)
pred_image = torch.zeros(img.shape)


device = ("cuda"
      if torch.cuda.is_available()
      else "mps"
      if torch.backends.mps.is_available()
      else "cpu")


net = FFN().to(device)
net.load_state_dict(torch.load(model_path))
net.eval()

dataset = SingleImageDataset(img_path + 'image.png')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

with torch.no_grad():
  for batch in dataloader:
    x, y, intensity = batch["x"], batch["y"], batch["intensity"]
    gt_image[:, y.item(), x.item()] = intensity*255

    x_ = x / dataset.w # normalizing x between 0 and 1
    y_ = y / dataset.h # normalizing y between 0 and 1

    coord = torch.stack((x_, y_), dim=-1)
    pred = net(coord.to(device))

    pred_image[:, y.item(), x.item()] = pred*255

pred_image = torch.clip(pred_image, 0, 255)

joint_image = torch.cat([gt_image.type(torch.uint8), pred_image.type(torch.uint8)], dim=2)
plt.imshow(joint_image.permute(1, 2, 0))
plt.axis('off')



channels, height, width = pred_image.shape

# initialize the outpainted image
outpainted_image = torch.zeros(channels, height + 2*20, width + 2*20)

for x in range(outpainted_image.shape[2]):
  for y in range(outpainted_image.shape[1]):

    x_ = torch.tensor((x - 20) / dataset.w)
    y_ = torch.tensor((y - 20) / dataset.h)

    coord = torch.stack((x_, y_), dim=-1)

    pred = net(coord.to(device))

    outpainted_image[:, y, x] = 255*pred

outpainted_image = torch.clip(outpainted_image, 0, 255).type(torch.uint8)

fig, ax = plt.subplots()
ax.imshow(outpainted_image.permute(1, 2, 0))
rect = patches.Rectangle((20, 20), dataset.w, dataset.h, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.axis('off')
plt.show()
plt.close()