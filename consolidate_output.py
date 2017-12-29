import imageio
import numpy as np
import torch
import torchvision
from fashion_mnist_gan import get_fashion_mnist

def make_gif(output_file='./training.gif',output_folder='./output/'):
    imgs = []
    for i in range(100):
        imgs.append(imageio.imread('{}output_{}.jpg'.format(output_folder,i)))
    imageio.mimsave(output_file, imgs)

def output_training_samples(output_file='./actuals.jpg',num_samples=200, num_rows=25, data=torchvision.datasets.FashionMNIST('./fashion_mnist')):
    inds = np.random.choice(np.arange(0, 60000), size=num_samples, replace=False)
    process_pil = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    deprocess_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
    tensors = [process_pil(data[ind][0]) for ind in inds]
    tensor = torch.cat(tensors, dim=0).view([num_samples,1,28,28])
    output = torchvision.utils.make_grid(tensor, nrow=num_rows)
    output = deprocess_pil(output)
    output.save(output_file)
    
if __name__=='__main__':
    make_gif()
    output_training_samples()
