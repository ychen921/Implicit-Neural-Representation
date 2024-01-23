# Implicit-Neural-Representation
In this project, we have hands-on experience with implicit neural representation (INR). With INR, we parameterize some signal (in our case images) with a neural network (feed-forward network). While in practice this might be useful for outpainting, super-resolution, and compression, in this project we will mainly focus on the basics, with some proof-of-concept outpainting at the end. Additionally, we apply sine activation function and gaussian fourier feature mapping to improve the reconstruction quality of the system.

## Dataset
The data that can be found in image data folder `bird.jpg`, we resize the image to 150 * 150  pixels in order to reduce the training time.

## Code Structure
`pre_process.py`: Resize the image to 150 * 150 pixels.

`data_reader.py`: A data loader that retrieves x & y coordinates and rgb values of each pixel.

`model.py`: A network of fully connected layer (FFN) which output the rgb values.

`train.ipynb`: putting all togather and train the model, also evaluate the model by PSNR.

`train_mapping.ipynb`: Same as `train.ipynb` but train the network with gaussian fourier feature mapping.

`train_Sine.ipynb`: Same as `train.ipynb` but train the network by using sine activation function.