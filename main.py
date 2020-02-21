import torch
import torchvision
import torchvision.transforms as transforms

from classes.FNN import FNN
from classes.CNN import CNN


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_batch_size = 10000
test_set = torchvision.datasets.FashionMNIST(
	root='./data',
	train=False,
	download=False,
	transform=transforms.Compose([transforms.ToTensor()]))
testloader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)


def to_mse_labels(targets):
	dataset_size = list(targets.size())[0]
	y_onehot = torch.FloatTensor(dataset_size, 10)
	y_onehot.zero_()
	for i, y in enumerate(targets):
		y_onehot[i][y] = 1
	return y_onehot


def print_file(model, loss, accuracy, file_name, is_mse=False):
	for data in testloader:
		inputs, labels = data

		if is_mse:
			inputs, labels = inputs.view(test_batch_size, -1).to(device), to_mse_labels(labels).to(device)
			_, labels = torch.max(labels, 1)
		else:
			inputs, labels = inputs.to(device), labels.to(device)

		outputs = model(inputs)
		_, pred = torch.max(outputs.data, 1)

		index = 0
		with open(file_name, 'w') as f:
			f.write("Loss on Test Data : %s\n" % loss)
			f.write("Accuracy on Test Data : %s\n" % accuracy)
			f.write("gt_label,pred_label\n")
			for true, predicted in zip(labels.tolist(), pred.tolist()):
				if index != len(labels) - 1:
					f.write("%s,%s\n" % (true, predicted))
				else:
					f.write("%s,%s" % (true, predicted))
				index += 1
	return


def convolution_neural_net():
	torch.manual_seed(0)
	model = CNN().to(device)
	model.load_state_dict(torch.load("./models/model_cnn", map_location=torch.device('cpu')))
	print_file(model=model, loss=0.104466, accuracy=0.9107, file_name="convolution_neural_net.txt")
	return


def multi_layer_net():
	torch.manual_seed(0)
	model = FNN().to(device)
	model.load_state_dict(torch.load("./models/model_fnn", map_location=torch.device('cpu')))
	print_file(model=model, loss=0.002703, accuracy=0.9033, file_name="multi_layer_net.txt", is_mse=True)
	return


convolution_neural_net()
multi_layer_net()


print("Name = Deepesh Virendra Hada")
print("IISc SR No. = 17196")
print("Course = M.Tech.")
print("Department = CSA")
