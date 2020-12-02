#include <stdio.h>
#include <stdlib.h>
#include <nnet.h>
#define EPOCHS 3
#define FALSE 0
#define TRUE 1

void write2file(double *buf, char *format, int dim, char *fileName);

int main(int argc, char *argv[])
{
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

    /* splitting the dataset in 'train' and 'validation' */
    char *selection = (char*)malloc(ds_train.dim);
    holdout(ds_train.dim, 0.10, selection); /* keeping 10% of data into the validation set */
    
    printf("\n* LEARNING\n\n");
    printf("Learning rate: %lf\n", net.learningRate);
    double validation_losses[EPOCHS];
    double min_loss = 1.0;
    for (int i = 0; i < EPOCHS; i++)
    {
        printf("Epoch %i/%i:\n", i + 1, EPOCHS);
        train(&net, &ds_train, selection);
        validate(&net, &ds_train, &validation_losses[i], selection, FALSE);

        /* keeping track of the model that gave the minimum loss */
        if (validation_losses[i] < min_loss)
        {
            min_loss = validation_losses[i];
            export_weights(&net, "weights.bin");    /* binary format */
            weights2text(&net, "weights.txt");      /* text format */
        }
    }
    
    /* logging all validation losses to a plain-text file */
    write2file(validation_losses, "%.8lf\n", EPOCHS, "val_losses.txt");
    printf(" Best loss: %.2lf%%\n", 100.0 * min_loss);

    free_dataset(&ds_train);
    free_net(&net);
    return 0;
}


void write2file(double *buf, char *format, int dim, char *fileName)
{
    FILE *f = fopen(fileName, "wt");
    for (int i = 0; i < dim; i++)
    {
        fprintf(f, format,buf[i]);
    }
    fclose(f);
}













