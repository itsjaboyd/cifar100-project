# CIFAR-100 Image Classification Project
Designing, training, and analyzing convolutional, artificial, and random forest networks
on the CIFAR-100 image classification dataset. This project makes use of TensorFlow and
its wrapper TFLearn for these neural network types.

### Accessing the Dataset
After researching and experimenting with the official downloadable dataset from 
[CS-Toronto](https://www.cs.toronto.edu/~kriz/cifar.html), turns out that TensorFlow
has prepackaged built-in datasets for users to explore, and CIFAR-100 happens to be
one of them. Instead of manually downloading the dataset, the project makes use of
the TensorFlow version, which makes the source code cleaner and easier to maintain. 

After running any of the drivers for the project, TensorFlow will attempt to download
the dataset directly into the directory the project resides and continue running normally.
The project does not look for any downloaded dataset files, so using the TensorFlow
version is the one and only option. The folder that TensorFlow creates is `cifar-100-python/`
so avoid modifying any files within this directory.

### Project Structure
The project is structured around each different type of neural network. The convolutional,
artificial, and random forest networks each have their own directory for source code:
`cnn/`, `ann/`, and `raf/` respectively. As mentioned before, TensorFlow's dataset source
files reside in `cifar-100-python/`. Each fully trained network resides in the `nets/`
directory, where the project takes each network for testing and statistical analysis.
The file `cifar.py` is the overarching driver for which the project is executed. Of course
this file `README.md` is for documentation and project details, and the project includes 
a `.gitignore` for version control purposes.

## Artificial Neural Networks

## Convolutional Neural Networks

## Randoom Forest Networks