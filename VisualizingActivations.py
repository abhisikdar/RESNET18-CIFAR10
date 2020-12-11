import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.models as models

import torchfunc #needs to be installed first

resnet18 =models.resnet18(pretrained=True, progress=True)

cifar_transformed = datasets.CIFAR10('/Users/abhinavasikdar/Desktop/ELL793 Computer Vision/Assignment3/',train=True,download=True, transform=transforms.Compose([
                                                                                                        transforms.Resize(256),transforms.CenterCrop(224),
                                                                                                        transforms.ToTensor()
]))

#For using a pretrained model on something other than ImageNet
#resnet18=torch.load('model.pt',map_location=torch.device('cpu'))


resnet18.eval()
recorder = torchfunc.hooks.recorders.ForwardPre()
recorder.modules(resnet18)

#Enumerating and printing all the modules in the network
for i, submodule in enumerate(resnet18.modules()):
    print(i, submodule)

#Checking which image to use for generating activation map
img, label = cifar_transformed[238] 
plt.imshow(img.permute(1, 2, 0))
plt.axis("off")
plt.show()


img=img.unsqueeze(0) #Since batch size = 1
out=resnet18(img) #Forward passing the image


conv_layer=2 #Which layer the activation maps are to be generated from +1
img_number=0 #Sequence position of the image in forward pass
img_obj='car'

#recorder.data[conv_layer][img_number].shape

img=recorder.data[conv_layer][img_number].squeeze() #Since batch size =1
img=img.detach().numpy() #Converting to numpy

#Vizualising and saving all the activation maps
for i in range(img.shape[0]):
    plt.imshow(img[i,:,:])
    plt.axis('off')
    plt.savefig(img_obj+'_'+'Conv_'+str(conv_layer)+'_'+str(i),bbox_inches='tight',pad_inches=0.0)
    plt.show()