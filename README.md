# CIFAR-100 Image Classification Project
Designing, training, and analyzing convolutional, artificial, and random forest networks
on the CIFAR-100 image classification dataset. This project makes use of TensorFlow and
its wrapper TFLearn as well as sklearn for training these neural network types. Project
source code can be found on [Jason Boyd's Github.](https://github.com/itsjaboyd/cifar100-project)

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
There were three planned files for each network type originally in the project proposal.
After thinking and designing out the project, a different route was taken in structure
to include multiple different files for each network type that included network designs,
driver code, and unit testing. The project is structured around each different type of 
neural network. The convolutional, artificial, and random forest networks each have their 
own directory for source code: `cnn/`, `ann/`, and `raf/` respectively. 

As mentioned before, TensorFlow's dataset source files reside in `cifar-100-python/`. Each 
fully trained network resides in the `nets/` directory, where the project takes each network 
for testing and statistical analysis. The file `cifar.py` is the overarching driver for which 
the project is executed. Of course this file `README.md` is for documentation and project 
details, and the project includes  a `.gitignore` for version control purposes.

Convolutional and artificial network functionality is shared between a common utility file
which is stored under the `utils/` directory. The utility file includes a base python file
that contains those shared functions and includes basic unit tests for those functions
as well. The `testing/` directory is purely meant for unit testing purposes in each network type.
This directly is only used when those tests are called, and to be disregarded when really
looking at the project as a whole.

### Project Report

The project report is saved as `report.pdf` and `report.docx` inside the top-most project
directory. The purpose of the project report is to show the findings, successes and failures, 
the program process, and reflections from building the project and training those networks.

## Running the Project
There exists an executive driver file `cifar.py` in the top level directory that ties together
all implementations from each subdirectory. There are multiple options to run the project, and
each method is explained below in detail, however the option intended for grading purposes and
viewing purposes is simply loading and testing the already supplied trained networks. 

1. **Loading and testing trained networks**

This method serves as the primary function of viewing this project. Upon calling this argument,
it fetches all saved networks and one by one performs the statistical analysis about them.
To run this standard project method, you supply running the file with 'standard' or no argument.
Please note that this project requires Python 3.6+ to run, so be sure to specify the python
command for which your installation resides (sometimes python3 ... or just regular python ...).
Running this command will also run all unit tests associated with each network type.

$ `python cifar.py` or `python cifar.py standard` 

2. **Miniature scale network creation and training**

This method serves as a secondary function for viewing this project. After calling this argument
from the command line, it builds simplified versions of each network architecture and trains each
for three epochs, then produces the statistics on each trained network. Note that this function
does not save the networks as this is more of a proof that the project works. Again, in order to
run this project method, the user must supply the file with the 'create' argument as such:

$ `python cifar.py create`

3. **Unit Testing Network Types**

This method serves as an addition to this project by running all unit tests from each type of
network at once. It was originally used just for testing purposes while building the project,
but as the project grew in size, this function was handy to keep around. Please note again that
all unit tests are run in addition to running the standard project method. To only run the unit
tests, run `cifar.py` while supplying the 'testing' argument, as such:

$ `python cifar.py testing`

## Convolutional Neural Networks

There are three versions of convolutional neural networks that are defined in the CNN subsection.
First, the best performing saved convolutional architecture is created by the `make_cifar_convnet()`
and loaded with the `load_cifar_convnet()` functions. The other two are meant for unit and other
testing purposes including an example and shallower convolutional architecture and created/loaded
functions in the same fashion as the best performing network. The example network comes straight
from the lecture notes and homework.

## Artificial Neural Networks

The artificial neural network subsection is very closely related to the convolutional portion.
There exist three different versions of artificial network structures with the best performing
saved artificial structure being created and loaded by the `make_cifar_artnet()` and `load_cifar_artnet()`
functions respectively. The other two exhibit larger and smaller artificial structures that are
created and loaded in again a similar fashion.

## Random Forest Networks

Unfortunately due to the time constraints and effort the rest of the project has taken in development
and training each network thus far, implementing the random forest section of this project could
not be reached. Everything was laid out for development, but again there was not enough time left to
experiment in that area. While the required data for analysis will not be presented in the project, there
will be estimations whereas random forests might compare to the trained convolutional and artificial
neural networks on CIFAR-100. Beyond the scope of grading and submission of this project, development
will most likely continue as results of the random forests would provide valuable insight into each
of the image classification network types.

## Dependencies and Versions

numpy -> 1.18.1\
scikit-learn -> 0.23.2\
sklearn -> 0.0\
tensorboard -> 1.15.0\
tensorflow -> 1.15.0\
tensorflow-estimator -> 1.15.1\
tflearn -> 0.3.2\

These are the dependencies and packages that I found that could apply to the project. I downloaded
them using `pip install package` where package is the package to be downloaded. To find the
comprehensive currently installed package list for your installation of Python 3, use the command
`pip freeze` or `pip3 freeze` depending on which command uses your Python 3 installation.