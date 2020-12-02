# Neural Network

This is a basic experimental implementation of a Fully-connected Multilayer Perceptron that
can be trained from scratch.
I created this library to have a feeling of how well a naive C implementation perform, so this is not
meant to be a complete.

## Structure

The model is not created specifying the linear transformations but rather specifying
the number of neurons for each layer. Once the layout is known, all the weights are stored **contiguously** in memory.

All the data relative to the values in the network is stored in a structure with 3 main arrays:
 - _stimulus_ contains the input of each neuron in a forward pass
 - _neurons_ contains the value of the stimulus after the application of the activation function
 - _sensitivity_ is proportional to the gradient and is propagated backward when training the net.

Each of these vectors is contiguous in memory, and is accessed by computing an offset w.r.t. a given layer.

## Data

The library provides functions only to process datasets in a plain-text format:
every row must contain all features separated by an ascii character and
class labels. Both regression and classification tasks can be carried out but
for the latter the label class must be encoded in one-hot format.

## Model creation example

This snippet creates a multilayer perceptron with 784, 100 and 10 neurons per layer
respectively, with _sig\_frac_ ( x/(1+|x|) )as activation function.

```c
Dataset ds_train = {
        .xdim = 784     /* number of input features */,
        .ydim = 10      /* number of classes */
        };
import_dataset(&ds_train,
    "data/train.dat" /* plain dataset file */,
    " "              /* features separator */,
    42000            /* number of elements to read */
    );

NeuralNet net;
int layout[] = { 784, 100, 10 }; /* number of neurons per layer (excluding bias) */
init_net(&net, layout,
    3               /* total number of layers (including input and output layers) */,
    &sig_frac       /* activation function */,
    &sig_frac_der   /* derivative of activation function */,
    &argmax         /* threshold function for the output layer */,
    1e-2            /* learning rate */
    );
print_net(&net);

/* random initial weights with uniform distribution in [-0.7; 0.7] */
set_rand_weights(&net, -0.7, +0.7);
```

## Execution example

The following is the output of _training.c_, that trains an NN for 3 epochs on
the partial [MNIST](http://yann.lecun.com/exdb/mnist/) dataset provided in the Kaggle [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) competition.

```
* IMPORT DATASET
Importing ...
Imported 42000 examples
* NEURAL NET
Structure               : Fully-connected Multilayer Perceptron
Number of layers        : 3
Number of weights       : 79510
Number of neurons       : 894
Neurons per layer       : { 784,100,10 } + biases
* VALIDATION: Hold out
 Dataset        : 42000 elements
 holdout set    : 4229 ( 10.07% )
 training set   : 37771 ( 89.93% )

* LEARNING

Learning rate: 0.010000
Epoch 1/3:
 Training:      100% [▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄] 6.6 sec
 Validating:    100% [▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄] 0.7 sec, loss: 21.33
Epoch 2/3:
 Training:      100% [▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄] 7.0 sec
 Validating:    100% [▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄] 0.8 sec, loss: 16.22
Epoch 3/3:
 Training:      100% [▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄] 6.9 sec
 Validating:    100% [▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄] 0.8 sec, loss: 14.26
 Best loss: 14.26%
```

