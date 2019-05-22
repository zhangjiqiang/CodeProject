import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

# 构造预加载模型的分类器
class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        if len(hidden_layers) > 1:
            layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
            self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        # self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        for each in self.hidden_layers:
            x = F.relu(each(x))
            # x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)

# ['vgg16','alexnet','resnet18','densenet121']
def generateModel(hidden_layer, arch='vgg16'):
    if arch == 'vgg16':
        train_model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        train_model = models.alexnet(pretrained=True)
    elif arch == 'resnet18':
        train_model = models.resnet18(pretrained=True)
    elif arch == 'densenet121':
        train_model = models.densenet121(pretrained=True)
    else:
        train_model = models.vgg16(pretrained=True)

    for param in train_model.parameters():
        param.requires_grad = False

    train_model.classifier = Classifier(25088, 102, hidden_layer)
    print("generateModel finished")
    return train_model

def validation(train_model, validloader, criterion, gpu=False):
    if gpu:
        train_model.cuda()
    else:
        train_model.cpu()

    valid_loss = 0
    accuracy = 0
    train_model.eval()
    with torch.no_grad():
        for images, labels in validloader:
            if gpu:
                images = images.cuda()
                labels = labels.cuda()
            output = train_model(images)  # 获取预测结果
            valid_loss += criterion(output, labels).item()
            equality = (labels == output.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        print("validSet loss:{:.4f}".format(valid_loss / len(validloader)),
              "validSet accuracy:{:.4f}".format(accuracy / len(validloader)))

def TestAccuracy(train_model, testloader, gpu=False):
    accuracy,test_loss = 0,0
    print("start comoute TestAccuracy:")
    if gpu:
        train_model.cuda()
    else:
        train_model.cpu()

        train_model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            if gpu:
                images = images.cuda()
                labels = labels.cuda()
            output = train_model(images) #获取预测结果
            equality = (labels == output.max(1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        print("test accuracy:{:.4f}".format(accuracy / len(testloader)))

def saveModel(save_path, train_model, train_data, optimizer, epochs, gpu=False, arch='vgg16'):
    if gpu:
        train_model.cuda()
    else:
        train_model.cpu()
    classifierPoint = {'input_size': 25088,
                       'output_size': 102,
                       'hidden_layers': [each.out_features for each in train_model.classifier.hidden_layers],
                       'dropout': 0.5}

    torch.save({"classifier": classifierPoint, "state_dict": train_model.state_dict(),
                "epoch": epochs, "optimizer_state_dict": optimizer.state_dict(),
                "class_ids_Map": train_data.class_to_idx,
                "arch": arch}, save_path)
    print("model saved")

def loadModel(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    if arch == 'vgg16':
        train_model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        train_model = models.alexnet(pretrained=True)
    elif arch == 'resnet18':
        train_model = models.resnet18(pretrained=True)
    elif arch == 'densenet121':
        train_model = models.densenet121(pretrained=True)
    else:
        train_model = models.vgg16(pretrained=True)

    train_model.classifier = Classifier(checkpoint['classifier']['input_size'],
                                  checkpoint['classifier']['output_size'],
                                  checkpoint['classifier']['hidden_layers'],
                                  checkpoint['classifier']['dropout'])
    train_model.load_state_dict(checkpoint['state_dict'])
    optimizer_t = optim.Adam(train_model.classifier.parameters(), lr=0.001)
    optimizer_t.load_state_dict(checkpoint['optimizer_state_dict'])
    class_ids_Map = checkpoint['class_ids_Map']
    print("load model finished")
    return train_model, optimizer_t, class_ids_Map



def loadData(data_dir):
    train_dir = 'train'
    valid_dir = 'valid'
    test_dir = 'test'
    train_transforms = transforms.Compose([transforms.RandomRotation(10),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_test_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(data_dir + train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + valid_dir, transform=valid_test_transforms)
    test_data = datasets.ImageFolder(data_dir + test_dir, transform=valid_test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    print("load data finished")
    return train_data, trainloader, validloader, testloader

def trainModel(train_model, trainloader, validloader, testloader, learning_rate, epochs, gpu=False):
    if train_model is not None:
        print("start trainning")
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(train_model.classifier.parameters(), lr=learning_rate)
        print_every = 40
        steps = 0
        train_loss = 0
        if gpu:
            train_model.cuda()
        else:
            train_model.cpu()
        for e in range(epochs):
            train_loss = 0
            train_model.train()
            for images, labels in trainloader:
                steps += 1
                labels = Variable(labels)
                images = Variable(images)
                if gpu:
                    labels = labels.cuda()
                    images = images.cuda()
                optimizer.zero_grad()
                outputs = train_model.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if steps % print_every == 0:
                    print("Epoch: {}/{}... ".format(e + 1, epochs),
                          "train_loss:{:.4f}".format(train_loss / print_every))
                    train_loss = 0
        
        validation(train_model, validloader, criterion, gpu)
        TestAccuracy(train_model, testloader, gpu)
        
        return train_model, optimizer