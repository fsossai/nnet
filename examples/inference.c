#include <stdio.h>
#include <stdlib.h>
#include <nnet.h>
#define FALSE 0
#define TRUE 1

int main(int argc, char *argv[])
{
    Dataset ds_test = {
        .xdim = 784     /* number of input features */,
        .ydim = 0       /* 0 for unclassified data */
        };
    import_dataset(&ds_test,
        "data/test.dat" /* plain dataset file */,
        ","             /* features separator */,
        28000           /* number of elements to read */
        );

    NeuralNet net;
    int layout[] = { 784, 100, 10 }; /* number of neurons per layer (excluding bias) */
    init_net(&net, layout,
        3               /* total number of layers (including input and output layers) */,
        &sig_frac       /* activation function */,
        &sig_frac_der   /* derivative of activation function */,
        &argmax         /* function for the output layer */,
        1e-3            /* learning rate */
        );
    
    /* Loading weights of a previously trained model */
    printf("* WEIGHTS LOADING\n");    
    if (import_weights(&net, "weights.bin"))
    {
        printf("ERROR: unable to import weights!\n");
        return -1;
    }
    else
        printf("Weights imported\n");
    
    print_net(&net);
    
    /* Inference on the test dataset */
    printf("\n* INFERENCE\n");
    double *predictions = (double*)malloc(sizeof(double) * ds_test.dim);
    classify(&net, &ds_test, predictions, NULL);

    /* Writing prediction to file */
    printf("Writing predictions to file ... ");
    FILE *f = fopen("predictions.csv","wt");

    fprintf(f,"ImageId,Label\n");
    for (int i = 0; i < ds_test.dim; i++)
        fprintf(f, "%i,%i\n", i+1, (int)predictions[i]);
    
    fclose(f);
    printf("Done\n");
    free(predictions);

    free_dataset(&ds_test);
    free_net(&net);
    return 0;
}













