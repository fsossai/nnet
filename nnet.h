
#define S_TRAIN 1
#define S_VALIDATION 0
#define LINE_MAX_DIM 4096
#define PROGRESS_BAR_SIZE 40
#define PROGRESS_BAR_CHAR 220

struct NeuralNet
{
    int T;                  // number of layers
    double *weights;
    double *neurons;
    double *stimulus;
    double *sensitivity;
    int *layerWidths;       // number of neurons for a given layer
    int netSize;            // total number of neurons
    int *neurOffset;        // position of the first neuron in the i-th layer, the last element equals to netSize;
    int *weightOffset;      // position of the first weight in the (i+1)-th layer (number of weights before i-th layer)
    int links;              // total number of weights
    double learningRate;       // learning rate in gradient descent used in backpropagation
    double (*act_fun)(double);        // pointer to the activation function
    double (*act_fun_der)(double);        // pointer to the derivative of the activation function
    void (*thresh_fun)(struct NeuralNet*);   // function to apply to the neurons in the output layer
};

struct Dataset
{
    double **x;
    double **y;
    int xdim;
    int ydim;
    int dim;
};

typedef struct Dataset Dataset;
typedef struct NeuralNet NeuralNet;

int argmax(NeuralNet *net);
void backward(NeuralNet *net);
void bin_thresh(NeuralNet *net, double threshold);
double compute_loss(NeuralNet *net, double *y);
void compute_sens(NeuralNet *net, double *y);
int export_weights(NeuralNet *net, char *filename);
void forward(NeuralNet *net);
void free_dataset(Dataset *ds);
void free_net(NeuralNet *net);
void holdout(int dim, double split, char *selection);
void init_net(NeuralNet *net, int *layerWidths, int T, double (*act_fun)(double),
             double (*act_fun_der)(double), void *thresh_fun, double learningRate);
void import_dataset(Dataset *ds, char *filename, char *splitter, int maxdim);
int import_weights(NeuralNet *net, char *filename);
void classify(NeuralNet *net, Dataset *ds, double *output, int (*format)(int));
void print_dataset(Dataset *ds);
void print_net(NeuralNet *net);
void print_neurons(NeuralNet *net);
void print_sens(NeuralNet *net);
void set_input(NeuralNet *net, double *x);
void set_rand_weights(NeuralNet *net, double min, double max);
void train(NeuralNet *net, Dataset *ds, char *selection);
void validate(NeuralNet *net, Dataset *ds, double *loss,
          char *selection, char output);
void weights2text(NeuralNet *net, char *filename);

/* ACTIVATION FUNCTIONS */
double sig_frac(double x);
double sig_frac_der(double x);
double sig_e(double x);
double sig_e_der(double x);
double sig_tanh(double x);
double sig_tanh_der(double x);
double relu(double x);
double relu_der(double x);
double identity(double x);
double identity_der(double x);
