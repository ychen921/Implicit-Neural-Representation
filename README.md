# Implicit Neural Representation
In this project, we have hands-on experience with implicit neural representation (INR). With INR, we parameterize some signal (in our case images) with a neural network (feed-forward network). While in practice this might be useful for outpainting, super-resolution, and compression, in this project we will mainly focus on the basics, with some proof-of-concept outpainting at the end. Additionally, we apply Sine activation function [2] and Gaussian Fourier feature mapping [1] to improve the reconstruction quality of the system.

## Dataset
For the data that can be found in the image data folder `bird.jpg`, we resized the image to 150 * 150  pixels in order to reduce the training time.

## Code Structure
`pre_process.py`: Resize the image to 150 * 150 pixels.

`data_reader.py`: A data loader that retrieves x & y coordinates and RGB values of each pixel.

`model.py`: A network of fully connected layers (FFN) that output the RGB values.

`train.ipynb`: putting all together and train the model, also evaluating the model by PSNR.

`train_mapping.ipynb`: Same as `train.ipynb` but train the network with Gaussian Fourier feature mapping.

`train_Sine.ipynb`: Same as `train.ipynb` but train the network by using Sine activation function.

## Performance
| Method       | PSNR (dB) |
| :---        |    :----:       | 
| Normal FCN  | 21.928  |
| Position Mapping| 32.796 |
| Sine Activation| |

## References
1. Tancik, Matthew, et al. "Fourier features let networks learn high frequency functions in low dimensional domains." Advances in Neural Information Processing Systems 33 (2020): 7537-7547.
2. Sitzmann, Vincent, et al. "Implicit neural representations with periodic activation functions." Advances in neural information processing systems 33 (2020): 7462-7473.
