import numpy as np
import torch





class DefenseMLPLegacy(torch.nn.Module):
    def __init__(self, out_size, kernel_size=7):
        super(DefenseMLPLegacy, self).__init__()

        # stride 1/2 -> pooling

        self.conv1 = torch.nn.Conv2d(3, kernel_size, 3)
        self.batchnorm1 = torch.nn.BatchNorm2d(kernel_size)
        self.relu = torch.nn.ReLU()

        # self.conv1 = torch.nn.Linear(input_size, hidden_size)
        # self.batchnorm1 = torch.nn.BatchNorm1d(hidden_size)
        # self.relu = torch.nn.ReLU()


        self.fc2 = torch.nn.Conv2d(kernel_size, 3, 3)
        self.batchnorm2 = torch.nn.BatchNorm2d(3)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(out_size, 1)


    def forward(self, x):

        hidden = self.conv1(x)
        batch1 = self.batchnorm1(hidden)
        relu = self.relu(batch1)



        hidden2 = self.fc2(relu)
        batch2 = self.batchnorm2(hidden2)
        relu2 = self.relu2(batch2)

        flattened = torch.flatten(relu2, start_dim=1)
        output = self.fc3(flattened)
        return output



class DefenseMLP(torch.nn.Module):
    def __init__(self, kernel_size=7, input_shape=( 64,64)):
        super(DefenseMLP, self).__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, kernel_size, 3),
            torch.nn.BatchNorm2d(kernel_size),
            torch.nn.ReLU(),

        # self.conv1 = torch.nn.Linear(input_size, hidden_size)
        # self.batchnorm1 = torch.nn.BatchNorm1d(hidden_size)
        # self.relu = torch.nn.ReLU()

            torch.nn.Conv2d(kernel_size, 3, 3),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU()
        )

        self.fc3 = torch.nn.Linear(self.compute_layer_size(input_shape), 1)

    def compute_layer_size(self, input_shape):
        input_shape = [1, 3] + list(input_shape)
        return np.prod(self.conv_layers(torch.zeros(input_shape)).size()[1:])

    def forward(self, x):

        # hidden = self.conv1(x)
        # batch1 = self.batchnorm1(hidden)
        # relu = self.relu(batch1)
        #
        #
        #
        # hidden2 = self.fc2(relu)
        # batch2 = self.batchnorm2(hidden2)
        # relu2 = self.relu2(batch2)

        relu2 = self.conv_layers(x)

        flattened = torch.flatten(relu2, start_dim=1)
        output = self.fc3(flattened)
        return output


class DefenseLinearA(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(DefenseLinearA, self).__init__()

        # self.conv1 = torch.nn.Conv2d(3, 7, 3)
        # self.batchnorm1 = torch.nn.BatchNorm2d(7)
        # self.relu = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.batchnorm1 = torch.nn.BatchNorm1d(hidden_size1)
        self.relu = torch.nn.ReLU()


        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.batchnorm2 = torch.nn.BatchNorm1d(hidden_size2)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_size2, 1)


    def forward(self, x):
        flattened = torch.flatten(x, start_dim=1)
        hidden = self.fc1(flattened)
        batch1 = self.batchnorm1(hidden)
        relu = self.relu(batch1)



        hidden2 = self.fc2(relu)
        batch2 = self.batchnorm2(hidden2)
        relu2 = self.relu2(batch2)


        output = self.fc3(relu2)
        return output


class DefenseLinear(torch.nn.Module):
    def __init__(self, input_size, hc1, hc2):
        super(DefenseLinear, self).__init__()

        # self.conv1 = torch.nn.Conv2d(3, 7, 3)
        # self.batchnorm1 = torch.nn.BatchNorm2d(7)
        # self.relu = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1)


    def forward(self, x):
        flattened = torch.flatten(x, start_dim=1)
        output = self.fc1(flattened)
        output = self.fc2(output)
        output = self.fc3(output)
        return output


class DefenseMaxPool(torch.nn.Module):
    def __init__(self, kernel_size=7, input_shape=( 64,64)):
        super(DefenseMaxPool, self).__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, kernel_size, 3),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.BatchNorm2d(kernel_size),
            torch.nn.ReLU(),

        # self.conv1 = torch.nn.Linear(input_size, hidden_size)
        # self.batchnorm1 = torch.nn.BatchNorm1d(hidden_size)
        # self.relu = torch.nn.ReLU()

            torch.nn.Conv2d(kernel_size, 3, 3),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU()
        )

        self.fc3 = torch.nn.Linear(self.compute_layer_size(input_shape), 1)

    def compute_layer_size(self, input_shape):
        input_shape = [1, 3] + list(input_shape)
        return np.prod(self.conv_layers(torch.zeros(input_shape)).size()[1:])

    def forward(self, x):

        # hidden = self.conv1(x)
        # batch1 = self.batchnorm1(hidden)
        # relu = self.relu(batch1)
        #
        #
        #
        # hidden2 = self.fc2(relu)
        # batch2 = self.batchnorm2(hidden2)
        # relu2 = self.relu2(batch2)

        relu2 = self.conv_layers(x)

        flattened = torch.flatten(relu2, start_dim=1)
        output = self.fc3(flattened)
        return output

class NormalizeModel(torch.nn.Module):
    def __init__(
            self,
            base_model, mean=(0.485, 0.456, 0.406) , std=(0.229, 0.224, 0.225)
    ):
        super(NormalizeModel, self).__init__()
        self.base = base_model
        self.mean = mean
        self.std = std
        # self.resize = nn.AdaptiveAvgPool2d((resize_shape[1], resize_shape[0]))

    def forward(self, img):
        img = torch.stack([F.normalize(i, self.mean, self.std) for i in img])
        # img = self.resize(img)
        x = self.base(img)  # x is a dict
        return x

