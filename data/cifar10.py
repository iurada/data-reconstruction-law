import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10


class FlatBinaryCIFAR10:
    def __init__(self, root='./datasets'):
        self.root = root
        self.total_examples = 50000
        self.input_dim = 3 * 32 * 32 # 3072
        self.train_split = CIFAR10(root=self.root, download=True, train=True, transform=T.Compose([T.ToTensor()]))

    def generate_data(self, n):
        neg, pos, selected_idxs = [], [], []
        for i in range(self.total_examples):
            x, y = self.train_split[i]
            if y == 6:
                neg.append(x.view(-1).unsqueeze(0).clone())
                selected_idxs.append(i)
            if y == 9:
                pos.append(x.view(-1).unsqueeze(0).clone())
                selected_idxs.append(i)
            if len(neg) == n//2 and len(pos) ==  n//2: 
                break

        X = torch.cat(neg[:n//2] + pos[:n//2])
        self.train_mean, self.train_std = X.mean(), X.std()
        X = (X - self.train_mean) / self.train_std
        Y = torch.tensor([-1.0 for _ in range(n//2)] + [1.0 for _ in range(n//2)])
        return X, Y


class FlatMulticlassCIFAR10: 
    def __init__(self, root='./datasets'):
        self.root = root
        self.total_examples = 50000
        self.input_dim = 3 * 32 * 32 # 3072
        self.train_split = CIFAR10(root=self.root, download=True, train=True, transform=T.Compose([T.ToTensor()]))

    def generate_data(self, n_examples_per_class=1, n_classes=4):
        assert n_examples_per_class > 0, '`n_examples_per_class` argument must be 1 at least.'
        assert n_classes in range(1, 10+1), '`n_classes` argument must be in range {1, ..., 10}'
        self.n_examples_per_class = n_examples_per_class
        self.n_classes = n_classes

        classes = {i: [] for i in range(n_classes)}
        for i in range(self.total_examples):
            x, y = self.train_split[i]
            if y in range(self.n_classes) and len(classes[y]) < n_examples_per_class:
                classes[y].append(x.view(-1).unsqueeze(0).clone())

            if all(len(classes[i]) == n_examples_per_class for i in range(n_classes)): 
                break
        
        X = torch.cat([torch.cat(classes[i]) for i in range(n_classes)])
        Y = torch.cat([torch.tensor([j for _ in range(n_examples_per_class)]) for j in range(n_classes)]).squeeze()
        self.train_mean, self.train_std = X.mean(), X.std()
        X = (X - self.train_mean) / self.train_std
        return X, Y
    

class BinaryCIFAR10: 
    def __init__(self, root='./datasets'):
        self.root = root
        self.total_examples = 50000
        self.input_dim = 3 * 32 * 32 # 3072
        self.train_mean, self.train_std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        self.train_split = CIFAR10(root=self.root, download=True, train=True, transform=T.Compose([T.Resize((32, 32)), 
                                                                                                   T.ToTensor(), 
                                                                                                   T.Normalize(self.train_mean, self.train_std)]))

    def generate_data(self, n):
        neg, pos, selected_idxs = [], [], []
        for i in range(self.total_examples):
            x, y = self.train_split[i]
            if y == 6:
                neg.append(x.unsqueeze(0))
                selected_idxs.append(i)
            if y == 9:
                pos.append(x.unsqueeze(0))
                selected_idxs.append(i)
            if len(neg) == n//2 and len(pos) == n//2: break

        X = torch.cat(neg[:n//2] + pos[:n//2])
        Y = torch.tensor([-1.0 for _ in range(n//2)] + [1.0 for _ in range(n//2)])
        return X, Y