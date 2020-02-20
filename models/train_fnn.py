import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pylab as pylab
from classes.FNN import FNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_mse_labels(targets):
	dataset_size = list(targets.size())[0]
	y_onehot = torch.FloatTensor(dataset_size, 10)
	y_onehot.zero_()
	for x, y in enumerate(targets):
		y_onehot[x][y] = 1
	return y_onehot


train_batch_size = 1000
train_set = torchvision.datasets.FashionMNIST(
	root='./data',
	train=True,
	download=True,
	transform=transforms.Compose([
		transforms.ToTensor()
	])
)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)

test_batch_size = 10000
test_set = torchvision.datasets.FashionMNIST(
	root='./data',
	train=False,
	download=True,
	transform=transforms.Compose([
		transforms.ToTensor()
	])
)
testloader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)


net = FNN().to(device)


def evaluation(dataloader, model=net):
	total, correct = 0, 0
	for data in dataloader:
		inputs, labels = data
		inputs, labels = inputs.view(batch_size, -1).to(device), to_mse_labels(labels).to(device)
		outputs = model(inputs)
		_, pred = torch.max(outputs.data, 1)
		total += labels.size(0)
		true = torch.argmax(labels, 1)
		correct += (pred == true).sum().item()
	return 100 * correct / total


def plot_loss(epochs, loss_arr):
	epoch_arr = list(range(1, epochs + 1))
	pylab.plot(epoch_arr, loss_arr, label = 'Loss vs Number of Epochs')

	tick_arr = [1, 30, 60, 90]
	pylab.xticks(tick_arr)

	pylab.legend(loc='upper right')
	pylab.xlabel("Number of Epochs")
	pylab.ylabel("Loss (Cross Entropy)")
	pylab.title('Feedforward Neural Network')
	pylab.savefig('plot_fnn.pdf')
	pylab.show()
	return


loss_function = nn.MSELoss()
opt = optim.Adam(net.parameters(), lr=0.0008)

loss_arr = []
loss_epoch_arr = []
max_epochs = 90

for epoch in range(max_epochs):
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data
		inputs, labels = inputs.view(batch_size, -1).float().to(device), to_mse_labels(labels).to(device)
		opt.zero_grad()
		outputs = net(inputs)
		loss = loss_function(outputs, labels)
		loss.backward()
		opt.step()

	loss_epoch_arr.append(loss.item())
	test_acc = evaluation(testloader)
	if test_acc >= 90:
		torch.save(net.state_dict(), "./model_fnn_best")
	print('Epoch: %d/%d, Test accuracy: %0.2f, Loss: %f' % (epoch + 1, max_epochs, test_acc, loss.item()))
print('Final Train Accuracy: %0.2f' % evaluation(trainloader))
torch.save(net.state_dict(), "./model_fnn")
plot_loss(epochs, loss_arr)
