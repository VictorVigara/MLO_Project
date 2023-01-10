from torch.utils.data import Dataset
import os
import torch

class MyDataset(Dataset): 
    def __init__(self, data_type, data_dir, transform):
        super().__init__()

        self.data_type = data_type
        self.data_dir = data_dir
        self.images, self.labels = self._load_data()
        self.transform = transform

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform is not None: 
            image = self.transform(image)

        return image, label
    def _load_data(self): 
        project_dir = os.getcwd()
        images_dir =  project_dir+"/"+self.data_dir+"/"+self.data_type+"_images.pt"
        labels_dir =  project_dir+"/"+self.data_dir+"/"+self.data_type+"_labels.pt"
        images = torch.load(images_dir)
        labels = torch.load(labels_dir)

        return images, labels