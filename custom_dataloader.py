import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


class FaceDataset(Dataset):
    def __init__(self, dic, data_root, transform=None):
        self.dic = dic
        self.keys = list(dic.keys())
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        cur_path = self.keys[idx]
        cur_label = self.dic[cur_path]
        full_path = self.data_root + "/" + cur_path
        # full_path = os.path.join(self.data_root, cur_path)
        img = Image.open(full_path)

        if not transforms:
            print("Set your transforms")
            raise ValueError

        x = self.transform(img)

        return x, cur_label


class FaceDataLoader:
    def __init__(self, batch_size, num_workers, path, txt_path):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = path
        self.text_path = txt_path

        self.train_image, self.test_image = self.load_train_test_list()

    @staticmethod
    def read_text_file(file_path):
        tmp = {}
        f = open(file_path, 'r')
        for line in f.readlines():
            line = line.replace('\n', '')
            split_line = line.split("\t")
            tmp[split_line[0]] = int(split_line[1])  # split[0] is image name and split[1] is class label

        return tmp

    def load_train_test_list(self):
        train_file_path = os.path.join(self.text_path, "train.txt")
        test_file_path = os.path.join(self.text_path, "test.txt")

        train_image = self.read_text_file(train_file_path)
        test_image = self.read_text_file(test_file_path)

        return train_image, test_image

    def run(self):
        train_loader = self.train()
        test_loader = self.test()

        return train_loader, test_loader

    def train(self):

        training_set = FaceDataset(dic=self.train_image,
                                   data_root=self.data_path,
                                   transform=transforms.Compose([
                                       transforms.Scale([250, 250]),
                                       transforms.ToTensor()
                                   ]))

        print('==> Training data :', len(training_set), ' image', training_set[1][0].size())

        train_loader = DataLoader(
            dataset=training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader

    def test(self):

        test_set = FaceDataset(dic=self.test_image,
                               data_root=self.data_path,
                               transform=transforms.Compose([
                                   transforms.Scale([250, 250]),
                                   transforms.ToTensor()
                               ]))

        print('==> Validation data :', len(test_set), ' image', test_set[1][0].size())

        val_loader = DataLoader(
            dataset=test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader
