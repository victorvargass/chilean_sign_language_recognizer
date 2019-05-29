from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets

from PIL import Image
import os

class Signs_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, img_path, transform = None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform
        self.img_path = img_path

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
    
    def __getclasses__():
        return list('ABCDEFHIKLMNOPQRTUVWY')

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get 
        image = Image.open(os.path.join(self.img_path,ID))
        #X = torch.tensor((np.asarray(image).T/255.).astype('float32'))
        #X = image
        #image = np.asarray(image)
        if self.transform:
            X = self.transform(image)
            
        y = self.labels[ID]
        return X, y
    
def get_image_transforms():
    # Image transformations
    image_transforms = {
        'train': transforms.Compose([
                    transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.2, hue=0),
                    #transforms.Grayscale(3),
                    transforms.RandomAffine(degrees = 5, translate=(0.2,0), fillcolor=(255,255,255)),
                    transforms.Resize(size=(100, 100)),
                    #transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    #transforms.CenterCrop(size=224),  # Image net standards
                    transforms.ToTensor(),
                    transforms.Normalize
                                (mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
                    #transforms.Grayscale(3),
                    transforms.Resize(size=(100, 100)),
                    #transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize
                                (mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ]),
        }
    return image_transforms