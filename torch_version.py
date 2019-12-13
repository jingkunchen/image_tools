import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import cv2
import numpy as np


# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 2   
BATCH_SIZE = 64
LR = 0.005         # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 5


train_input = torch.from_numpy(np.load("/Users/chenjingkun/Documents/data/texture/uiuc_texture_painting_train.npy")).float()
train_output = torch.from_numpy(np.load("/Users/chenjingkun/Documents/data/texture/uiuc_texture_train_double.npy")).float()
# print(train_input.size())     
# print(train_output.size())   
# cv2.imshow('input_image', train_input[2].numpy())
# cv2.waitKey(0)
class subDataset(Data.dataset.Dataset):
    def __init__(self, Input, Output):
        self.Input = Input
        self.Output = Output
    def __len__(self):
        return len(self.Input)
    def __getitem__(self, index):
        input = torch.Tensor(self.Input[index])
        output = torch.Tensor(self.Output[index])
        return input, output
    def __getitems__(self, start, end):
        inputs = torch.Tensor(self.Input[start:end])
        outputs = torch.Tensor(self.Output[start:end])
        return inputs, outputs

dataset = subDataset(train_input, train_output)

# print('dataset大小为：', dataset.__len__())
# print(dataset.__getitem__(0))
# print(dataset[0][0])
# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
# deal_dataset = Data.TensorDataset(train_input, train_output)
train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 2)

# for i, item in enumerate(train_loader):
#         print('i:', i)
#         input, output = item
#         print('data:', input.shape)
#         print('label:', output.shape)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(224*224, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 224*224),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()
print(autoencoder)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()   # continuously plot

# original data (first row) for viewing
view_data, _ = dataset.__getitems__(0, N_TEST_IMG)
for i in range(N_TEST_IMG):
    print(view_data[i].numpy().shape)
    a[0][i].imshow(view_data[i].numpy().reshape(224, 224), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

for epoch in range(EPOCH):
    for step, (x, b_label) in enumerate(train_loader):
        b_x = x.view(-1, 224*224)   # batch x, shape (batch, 28*28)
        b_y = x.view(-1, 224*224)   # batch y, shape (batch, 28*28)

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

            # plotting decoded image (second row)
            print(view_data.shape)
            _, decoded_data = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (224, 224)), cmap='gray')
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()

# visualize in 3D plot
view_data = train_data.train_data[:200].view(-1, 224*224).type(torch.FloatTensor)/255.
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2); ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_data.train_labels[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
plt.show()