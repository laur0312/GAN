# GAN
Tensorflow implementation of some GANs.

## Results for mnist
The following results can be reproduced with command:  
```
python main.py --model <TYPE>
```

*Name* | *Epoch 1* | *Epoch 10* | *Epoch 25* |
:---: | :---: | :---: | :---: |
GAN | <img src='results/GAN/00.png' height='200px'> | <img src='results/GAN/09.png' height='200px'> | <img src='results/GAN/24.png' height='200px'> |
DCGAN | <img src='results/DCGAN/00.png' height='200px'> | <img src='results/DCGAN/09.png' height='200px'> | <img src='results/DCGAN/24.png' height='200px'> |
CDCGAN | <img src='results/CDCGAN/00.png' height='200px'> | <img src='results/CDCGAN/09.png' height='200px'> | <img src='results/CDCGAN/24.png' height='200px'> |
WGAN | <img src='results/WGAN/00.png' height='200px'> | <img src='results/WGAN/09.png' height='200px'> | <img src='results/WGAN/24.png' height='200px'> |

## Acknowledgements
This implementation has been tested with Tensorflow over ver1.3 and Python3 on Windows 10.