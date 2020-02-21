import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pylab as pylab
from classes.CNN import CNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transformations = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])

train_batch_size = 1000
train_set = torchvision.datasets.FashionMNIST(
	root='./data',
	train=True,
	download=True,
	transform=data_transformations
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


def evaluation(dataloader, model, build_confusion_matrix=False):
	total, correct = 0, 0
	classes = 10
	confusion_matrix = torch.zeros(classes, classes)
	for data in dataloader:
		inputs, labels = data
		inputs, labels = inputs.to(device), labels.to(device)
		outputs = model(inputs)
		_, pred = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (pred == labels).sum().item()

		if build_confusion_matrix:
            confusion = confusion_matrix(labels.view(-1), pred.view(-1))
            print(confusion)

	return 100 * correct / total


def plot_loss(epochs, loss_arr):
	epoch_arr = list(range(1, epochs + 1))
	pylab.scatter(epoch_arr, loss_arr)
	pylab.plot(epoch_arr, loss_arr, label='Loss vs Number of Epochs')

	tick_arr = [1, 30, 60, 90]
	pylab.xticks(tick_arr)
	pylab.legend(loc='upper right')
	pylab.xlabel("Number of Epochs")
	pylab.ylabel("Loss (Cross Entropy)")
	pylab.savefig('plot_cnn.pdf')
	pylab.show()
	return


net = CNN().to(device)
loss_function = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters(), lr=0.0008)
loss_arr = []
loss_epoch_arr = []
max_epochs = 90
loss = 0
for epoch in range(max_epochs):
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data
		inputs, labels = inputs.to(device), labels.to(device)
		opt.zero_grad()
		outputs = net(inputs)
		loss = loss_function(outputs, labels)
		loss.backward()
		opt.step()
	loss_epoch_arr.append(loss.item())
	print('Epoch: %d/%d, Test accuracy: %0.2f, Loss: %f' % (epoch + 1, max_epochs, evaluation(testloader), loss.item()))
print('Final Train Accuracy: %0.2f', (evaluation(trainloader)))

torch.save(net.state_dict(), "./model_cnn")
plot_loss(max_epochs, loss_epoch_arr)
