#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <nnet.h>

/* applies a transformation to the output layer: 
    the maximum value is set to 1.0 and the others to 0.0.
    Returns the index of the value set to 1.0 */
int argmax(NeuralNet *net)
{
    double max = net->neurons[net->neurOffset[net->T-1] + 0];
    int index = 0;
    for (int i = 1; i < net->layerWidths[net->T-1]; i++)
    {
        if (net->neurons[net->neurOffset[net->T-1] + i] > max)
        {
            max = net->neurons[net->neurOffset[net->T-1] + i];
            index = i;
        }
    }
    for (int i = 0; i < net->layerWidths[net->T-1]; i++)
        net->neurons[net->neurOffset[net->T-1] + i] = 0.0;
    net->neurons[net->neurOffset[net->T-1] + index] = 1.0;
    return index;
}

void backward(NeuralNet *net)
{
    for (int t = 1; t < net->T; t++)
    {
        for (int i = 0; i < net->layerWidths[t-1]; i++)
        {
            for (int j = 0; j < net->layerWidths[t]; j++)
            {
                net->weights[net->weightOffset[t-1] + i * net->layerWidths[t] + j] -=
                        net->learningRate * net->neurons[net->neurOffset[t-1] + i]
                        * net->sensitivity[net->neurOffset[t] + j];
            }
        }
        /* backpropagating on biases */
        for (int j = 0; j < net->layerWidths[t]; j++)
        {
            net->weights[net->weightOffset[t-1] + net->layerWidths[t-1] * net->layerWidths[t] + j] -=
                    net->learningRate * net->sensitivity[net->neurOffset[t] + j];
        }
    }
}

void bin_thresh(NeuralNet *net, double threshold)
{
    for (int i = 0; i < net->layerWidths[net->T-1]; i++)
    {
        net->neurons[net->neurOffset[net->T-1] + i] =
            (net->neurons[net->neurOffset[net->T-1] + i] > threshold) ? 1 : 0;
    }
}

void classify(NeuralNet *net, Dataset *ds, double *output, int (*format)(int))
{
    printf(" Inference:\t000%% [");
    double timer = -clock();
    const int N = ds->dim;
    const int step = N / 100;
    for (int i = 0; i < N; i++)
    {
        
        set_input(net, ds->x[i]);
        forward(net);
        if (net->thresh_fun != NULL)
            net->thresh_fun(net);
        for (int j = 0; j < net->layerWidths[net->T-1]; j++)
        {
            if ( net->neurons[net->neurOffset[net->T-1] + j] == 1.0 )
            {
                output[i] = (format) ? format(j) : j;
                break;
            }
        }

        if (i % step == 0)
        {
            printf("\r Inference:\t%03.0f%% [", 100.0 * (float)i/(float)N);
            const int bar_chars = (int)((float)i/(float)N * PROGRESS_BAR_SIZE);
            const int blankets = PROGRESS_BAR_SIZE - bar_chars;
            for (int j = 0; j < bar_chars; j++)
                printf("%c", PROGRESS_BAR_CHAR);
            for (int j = 0; j < blankets; j++)
                printf(" ");
            printf("] ");
        }
    }
    timer += clock();
    timer /= CLOCKS_PER_SEC;
    printf("\r Inference:\t100%% [");
    for (int j = 0; j < PROGRESS_BAR_SIZE; j++)
        printf("%c", PROGRESS_BAR_CHAR);
    printf("] %.1lf sec\n", timer);
}

double compute_loss(NeuralNet *net, double *y)
{
    return !y[argmax(net)];
}

void compute_sens(NeuralNet *net, double *y)
{
    double delta, sens, partialSum;
    for (int i = 0; i < net->layerWidths[net->T-1]; i++)
    {
        delta = net->neurons[net->neurOffset[net->T-1] + i] - y[i];
        sens = net->act_fun_der( net->stimulus[ net->neurOffset[net->T-1] + i ] );
        net->sensitivity[net->neurOffset[net->T-1] + i] = delta * sens;
    }
    /* computing backward */
    for (int t = net->T - 2; t > 0; t--)
    {
        for (int i = 0; i < net->layerWidths[t]; i++)
        {
            partialSum = 0;
            for (int j = 0; j < net->layerWidths[t+1]; j++)
            {
                partialSum +=
                    net->weights[net->weightOffset[t] + i * net->layerWidths[t+1] + j]
                    * net->sensitivity[net->neurOffset[t+1] + j];
            }
            net->sensitivity[net->neurOffset[t] + i] =
                        net->act_fun_der(net->stimulus[ net->neurOffset[t] + i ])
                        * partialSum;
        }
    }
}

int export_weights(NeuralNet *net, char *filename)
{
    FILE *f = fopen(filename, "wb");
    if (!f)
        return -1;
    if (fwrite(net->weights, sizeof(double), net->links, f))
    {
        fclose(f);
        return -2;
    }
    fclose(f);
    return 0;
}

void forward(NeuralNet *net)
{
    for (int t = 1; t < net->T; t++)
    {
        for (int i = 0; i < net->layerWidths[t-1]; i++)
        {
            net->neurons[net->neurOffset[t-1] + i] = net->act_fun(net->stimulus[net->neurOffset[t-1] + i]);
            for (int j = 0; j<net->layerWidths[t]; j++)
            {
                net->stimulus[net->neurOffset[t] + j] +=
                        net->weights[net->weightOffset[t-1] + i * net->layerWidths[t] + j] *
                        net->neurons[net->neurOffset[t-1] + i];
            }
        }
        /* computing bias */
        for (int j = 0; j < net->layerWidths[t]; j++)
        {
            net->stimulus[net->neurOffset[t] + j] +=
                    net->weights[net->weightOffset[t-1] + net->layerWidths[t-1] * net->layerWidths[t] + j];
        }
    }
    /* computing activation functions for the output layer */
    for (int i = 0; i < net->layerWidths[net->T-1]; i++)
    {
        net->neurons[net->neurOffset[net->T-1] + i] = net->act_fun(net->stimulus[net->neurOffset[net->T-1] + i]);
    }
}

void free_dataset(Dataset *ds)
{
    for (int i = 0; i < ds->dim; i++)
    {
        free(ds->x[i]);
        free(ds->y[i]);
    }
    free(ds->x);
    free(ds->y);
}

void free_net(NeuralNet *net)
{
    free(net->layerWidths);
    free(net->neurOffset);
    free(net->neurons);
    free(net->sensitivity);
    free(net->stimulus);
    free(net->weightOffset);
    free(net->weights);
}

/* selection[i] will be set to S_TRAIN (resp. S_VALIDATION) if the
    i-th element is selected as a training element (resp. validation element)
*/
void holdout(int dim, double split, char *selection)
{
    int counter = 0;
    double ratio;
    srand(time(0));
    for (int i = 0; i<dim; i++)
    {
        selection[i] = (( (double)rand() / (double)RAND_MAX ) < split) ? S_VALIDATION : S_TRAIN;
        counter += selection[i];
    }
    ratio = (double)counter / (double)dim;
    printf("* VALIDATION: Hold out\n");
    printf(" Dataset\t: %i elements\n",dim);
    printf(" holdout set\t: %i ( %.2f%% )\n", dim - counter, 100 - ratio * 100);
    printf(" training set\t: %i ( %.2f%% )\n", counter, ratio * 100);
}

/* memory initialization of the network */
void init_net(NeuralNet *net, int *layerWidths, int T, double (*act_fun)(double),
             double (*act_fun_der)(double), void *thresh_fun, double learningRate)
{
    net->T = T;
    net->learningRate = learningRate;
    net->layerWidths = (int*)malloc(sizeof(int) * T);
    net->neurOffset = (int*)malloc(sizeof(int) * (T + 1));
    net->weightOffset = (int*)malloc(sizeof(int) * T);
    net->netSize = 0;
    net->weightOffset[0] = 0;
    net->neurOffset[0] = 0;
    net->act_fun = act_fun;
    net->act_fun_der = act_fun_der;
    net->thresh_fun = thresh_fun;
    memcpy(net->layerWidths, layerWidths, sizeof(int) * T);
    
    for (int i = 0; i<T; i++)
    {
        net->netSize += layerWidths[i];
        net->neurOffset[i+1] = net->neurOffset[i] + layerWidths[i];
    }
    for (int i = 1; i<T; i++)
        net->weightOffset[i] = net->weightOffset[i-1]
                    + (net->layerWidths[i-1] + 1) * net->layerWidths[i];
    net->links = net->weightOffset[T-1];
    net->stimulus = (double*)calloc(sizeof(double), net->netSize);
    net->weights = (double*)calloc(sizeof(double), net->links);
    net->neurons = (double*)calloc(sizeof(double), net->netSize);
    net->sensitivity = (double*)calloc(sizeof(double), net->netSize);
    srand( time(0) );
}

/* a dataset is loaded into the main memory */
void import_dataset(Dataset *ds, char *filename, char *splitter, int maxdim)
{
    printf("* IMPORT DATASET\n");
    
    FILE *f = fopen(filename,"rt");
    if (!f)
    {
        fprintf(stderr, "No such file ( \"%s\" )\n",filename);
        return;
    }
    printf("Importing ...\n");
    fflush(stdout);
    double cur;
    int row = 0;
    int xpos = 0;
    int ypos = 0;
    int xdim = ds->xdim;
    int ydim = ds->ydim;
    double **x = (double**)malloc(sizeof(double*) * maxdim);
    double **y = (double**)malloc(sizeof(double*) * maxdim);
    ds->x = x;
    ds->y = y;
    x[0] = (double*)malloc(sizeof(double) * xdim);
    y[0] = (double*)malloc(sizeof(double) * ydim);
    char *line = (char*)malloc(LINE_MAX_DIM + 1);
    char *currentNum = (char*)malloc(128 + 1);
    while ( row < maxdim && fgets(line,LINE_MAX_DIM,f) )
    {
        currentNum = strtok(line, splitter);
        do
        {
            sscanf(currentNum, "%lf", &cur);
            if (xpos < xdim)
            {
                x[row][xpos++] = cur;
            }
            else if (ypos < ydim)
            {
                y[row][ypos++] = cur;
            }
            if (xpos == xdim && ypos == ydim)
            {
                row++;
                x[row] = (double*)malloc(sizeof(double) * ds->xdim);
                y[row] = (double*)malloc(sizeof(double) * ds->ydim);
                xpos = ypos = 0;
                currentNum = strtok(0, splitter);
            }
        }
        while ( (currentNum = strtok(0, splitter)) );
    }
    ds->dim = row;
    fclose(f);
    printf("Imported %i examples\n",row);
}

/* imports the weights of a previously trained model that has the same network structure */
int import_weights(NeuralNet *net, char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
        return -1;
    int r = fread(net->weights, sizeof(double), net->links, f);
    if (r < net->links)
    {
        fclose(f);
        return -2;
    }
    fclose(f);
    return 0;
}

void print_net(NeuralNet *net)
{
    printf("* NEURAL NET\n");
    printf("Structure\t\t: Fully-connected Multilayer Perceptron\n");
    printf("Number of layers\t: %i\n",net->T);
    printf("Number of weights\t: %i\n",net->links);
    printf("Number of neurons\t: %i\n",net->netSize);
    printf("Neurons per layer\t: { ");
    for (int t = 0; t < net->T; t++)
    {
        printf("%i%c",net->layerWidths[t],(t!=net->T-1)?',':'\0');
    }
    printf("} + biases\n");
}

void print_dataset(Dataset *ds)
{
    printf("* DATASET\n");
    int i;
    printf("\t");
    for (i = 0; i < ds->xdim; i++)
        printf("[x%i]\t",i);
    printf("||\t");
    for (i = 0; i < ds->ydim; i++)
        printf("[y%i]\t",i);
    printf("\n\t");
    for (i = 0; i < ds->xdim; i++)
        printf("\t");
    printf("||\t");
    for (i = 0; i < ds->ydim; i++)
        printf("\t");
    printf("\n");

    for (int r = 0; r < ds->dim; r++)
    {
        printf("[%.4i]\t", r+1);
        for (i = 0; i < ds->xdim; i++)
            printf("%.2lf\t", ds->x[r][i]);
        printf("||\t");
        for (i = 0; i < ds->ydim; i++)
            printf("%.2lf\t", ds->y[r][i]);
        printf("\n");
    }

    printf("\n%i elements\n\n", ds->dim);
}

void print_neurons(NeuralNet *net)
{
    printf("* NEURONS ACTIVATIONS\n");
    for (int t = 0; t < net->T; t++)
    {
        printf("layer %i:\n { ",t);
        for (int i = 0; i<net->layerWidths[t]; i++)
            printf( (i != net->layerWidths[t] - 1) ? "%.4f, " : "%.4f }\n",
                   net->neurons[net->neurOffset[t] + i] );
    }
}

void print_sens(NeuralNet *net)
{
    printf("* SENSITIVITIES\n");
    for (int t = 0; t < net->T; t++)
    {
        printf("layer %i:\n { ",t);
        for (int i = 0; i<net->layerWidths[t]; i++)
            printf( (i != net->layerWidths[t] - 1) ? "%.2f, " : "%.2f }\n",
                   net->sensitivity[net->neurOffset[t] + i] );
    }
}

/* sets the element 'x' as an input to the network. */
void set_input(NeuralNet *net, double *x)
{
    memset(net->stimulus, 0, sizeof(double) * net->netSize);
    memset(net->neurons, 0, sizeof(double) * net->netSize);
    for (int i = 0; i < net->layerWidths[0]; i++)
        net->stimulus[i] = x[i];
}

/* weights are initialized according to a uniform probability distribution 
    between 'min' and 'max' */
void set_rand_weights(NeuralNet *net, double min, double max)
{
    double cur;
    for (int i = 0; i < net->links; i++)
    {
        cur = (double)rand() / (double)RAND_MAX;
        cur *= (max - min);
        net->weights[i] = min + cur;
    }
}

void train(NeuralNet *net, Dataset *ds, char *selection)
{
    printf(" Training:\t000%% [");
    double timer = -clock();
    const int N = ds->dim;
    const int step = N / 100;
    for (int i = 0; i < N; i++)
    {
        if (selection[i] == S_TRAIN)
        {
            set_input(net, ds->x[i]);
            forward(net);
            compute_sens(net, ds->y[i]);
            backward(net);
        }

        if (i % step == 0)
        {
            printf("\r Training:\t%03.0f%% [", 100.0 * (float)i/(float)N);
            const int bar_chars = (int)((float)i/(float)N * PROGRESS_BAR_SIZE);
            const int blankets = PROGRESS_BAR_SIZE - bar_chars;
            for (int j = 0; j < bar_chars; j++)
                printf("%c", PROGRESS_BAR_CHAR);
            for (int j = 0; j < blankets; j++)
                printf(" ");
            printf("] ");
        }
    }
    timer += clock();
    timer /= CLOCKS_PER_SEC;
    printf("\r Training:\t100%% [");
    for (int j = 0; j < PROGRESS_BAR_SIZE; j++)
        printf("%c", PROGRESS_BAR_CHAR);
    printf("] %.1lf sec\n", timer);
}

void validate(NeuralNet *net, Dataset *ds, double *loss,
          char *selection, char output)
{
    double timer = -clock();
    double avgloss = 0;
    int k = 0;
    FILE *f = NULL;
    if (output)
        f = fopen("output.txt","wt");
    
    const int N = ds->dim;
    const int step = N / 100;
    printf(" Validating:\t000%% [");
    for (int i = 0; i < N; i++)
    {
        if (selection[i] == S_VALIDATION)
        {
            set_input(net, ds->x[i]);
            forward(net);
            if (output)
            {
                for (int j = 0; j < 10; j++)
                    fprintf(f,"%.5lf ",net->neurons[net->neurOffset[net->T-1] + j]);
                fprintf(f,"\r\n");
            }
            if (net->thresh_fun != NULL)
                net->thresh_fun(net);
            avgloss += compute_loss(net, ds->y[i]);
            k++;
        }
        
        if (i % step == 0)
        {
            printf("\r Validating:\t%03.0f%% [", 100.0 * (float)i/(float)N);
            const int bar_chars = (int)((float)i/(float)N * PROGRESS_BAR_SIZE);
            const int blankets = PROGRESS_BAR_SIZE - bar_chars;
            for (int j = 0; j < bar_chars; j++)
                printf("%c", PROGRESS_BAR_CHAR);
            for (int j = 0; j < blankets; j++)
                printf(" ");
            printf("] ");
        }
    }
    if (output)
        fclose(f);
    *loss = avgloss / k;

    timer += clock();
    timer /= CLOCKS_PER_SEC;
    printf("\r Validating:\t100%% [");
    for (int j = 0; j < PROGRESS_BAR_SIZE; j++)
        printf("%c", PROGRESS_BAR_CHAR);
    printf("] %.1lf sec, ", timer);
    printf("loss: %.2lf\n", 100.0 * (*loss));
}

void weights2text(NeuralNet *net, char *filename)
{
    FILE *f = fopen(filename, "wt");
    if (!f)
        return;
    else
    {
        for (int i = 0; i < net->links; i++)
        {
            fprintf(f, "%.8lf\n", net->weights[i]);
        }
    }
    fclose(f);
}


/* ACTIVATION FUNCTIONS */

double sig_frac(double x)
{
    return x / (1 + fabs(x));
}

double sig_frac_der(double x)
{
    return 1 / pow( ( 1 + fabs(x) ), 2 );
}

double sig_e(double x)
{
    return 1 / (1 + exp(-x));
}

double sig_e_der(double x)
{
    return exp(-x) / pow( 1 + exp(-x), 2 );
}

double sig_tanh(double x)
{
    return tanh(x);
}

double sig_tanh_der(double x)
{
    return 1 / pow(cosh(x), 2);
}

double relu(double x)
{
    return x > 0 ? x : 0;
}

double relu_der(double x)
{
    return x > 0 ? 1 : 0;
}

double identity(double x)
{
    return x;
}

double identity_der(double x)
{
    return 1.0;
}


