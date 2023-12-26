import torch
from numpy.ma.core import size
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import FFN
from data_reader import SingleImageDataset
from INR_pos_encod import pose_encod

img_path = r'data/'

device = ("cuda"
      if torch.cuda.is_available()
      else "mps"
      if torch.backends.mps.is_available()
      else "cpu")

print(f"Using {device} device")

####################################
m = 100
B = 10 * torch.randn(m, 2).to(device)
####################################


net = FFN(input_dim=2 * m).to(device)

lr = 1e-4
b_size = 128

dataset = SingleImageDataset(img_path + 'image.png')
dataloader = DataLoader(dataset, batch_size=b_size, shuffle=False)

# We are training the network for pixels,
# so will do a pixelwise MSE loss
criterion = torch.nn.MSELoss()

# Optimizer and number of epochs
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
NUM_EPOCHS = 1000

loss_values = []

# train and store loss values
for epoch in range(NUM_EPOCHS):
  for batch in dataloader:
      x, y, actual = batch["x"], batch["y"], batch["intensity"]
      x = x / dataset.w # normalizing x between 0 and 1
      y = y / dataset.h # normalizing y between 0 and 1
      
      coord = torch.stack((x, y), dim=-1).to(device)
      
      pose_coord = pose_encod(B, coord)
      
      ### Assemble coord from x and y, pass to net, compute loss
      pred = net(pose_coord)
      loss = criterion(pred, actual.float().to(device))
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  # Track loss
  loss_values.append(loss.item())
  avg_loss = sum(loss_values) / len(loss_values)
  print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss}\n")

torch.save(net.state_dict(), img_path+'pos_encode_model.pt')
plt.plot(range(1, NUM_EPOCHS + 1), loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.show()
