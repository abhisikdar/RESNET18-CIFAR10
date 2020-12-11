# RESNET18-CIFAR10
PyTorch code for ResNet18 on CIFAR10+Tiny CIFAR10 w/ Augmentations + Transfer Learning + Activation Maps
## Disclaimer:
This assignemnt is being done by
  1) Abhinava Sikdar 2017MT10724 www.github.com/abhisikdar
  2) Yashank Singh 2017MT10756 www.github.com/alyashgo

We take full responsibility of the code provided.

## About:
We use PyTorch for all the four parts.
  1) Training ResNet18 from scratch with CIFAR-10
  2) Using pretrained ResNet18 on ImageNet for CIFAR-10 with variations such as <br />
    a) Fine tuning the single and last FC layer <br />
    b) Using and fine tuning two FC layers <br />
    c) Deep Gradients
  3) Use Tiny CIFAR-10 with augmentations and dropout layers. We use the standard MoCoV2 augmentations here.
  4) Visualize the activation maps for all the above cases for initial, middle and last Conv2D layers.
  
