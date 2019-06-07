/* 
Author: Sujoy Purkayastha 
random function generator adopted from: http://www.cs.utsa.edu/~wagner/CS2073/random/random.html
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* A struct that represents the Neurons within a neural network */
typedef struct Neuron{ 
    float value;                  // the value of a Neuron struct  
    float* weightsPtr;            // a pointer to a float array of weights corresponding to each next neuron 
    float bias;                   // the bias of a node 
} Neuron;

/* A struct that contains a pointer to a contiguous block of Neuron structs in memory */
/* Will be used to make interfacing layers of Neurons easier */
typedef struct neuralLayer{
    Neuron* layerPtr;
    int numNeurons;
    struct neuralLayer* prevLayer;
    struct neuralLayer* nextLayer;
} neuralLayer; 

/* A struct that encapsulates an entire Neural Network model */
typedef struct neuralNet{
    neuralLayer* inputLayer;
    int numInputNeurons;
    int numHiddenLayers;
    int numNeuronsPerHiddenLayer;
    int numOutputNeurons;
} neuralNet;

/* A function that returns the value of a sigmoid function at a particular x-value */ 
float sigmoid(float x){
    return 1/(1 + exp(-x));
}

/* A function that normalizes the numerics that will be utilized by the neural network */
float normalize(float x, float xMin, float xMax){
    return (x - xMin)/(xMax - xMin);
}


double randDouble() {
   return rand()/(double)RAND_MAX;
}

double randDouble2(double x, double y) {
   return (y - x)*randDouble() + x;
}
 
void weightInitialization(float* iterator, int numNeuronsPerLayer){
    //printf("Weights: ");
    for(int i = 0; i < numNeuronsPerLayer; i++){
        *iterator = randDouble2(-1, 1); //initialize the weight b/w -1 and 1
        //printf("%f ", *iterator);
        iterator++;
    }
}

/* A function that simplifies the process of making Neuron layers */
/* After this function is executed, a pointer pointing to a neuralLayer struct will be returned */
neuralLayer* layerInitialization(int numNeurons, int numNeuronsNextLayer){
    Neuron* LayerPtr = malloc(sizeof(Neuron) * numNeurons);
    Neuron* tempPtr = LayerPtr; //temp pointer to iterate through the new block of Neurons 
    //initialize the input layer structs 
    for(int i = 0; i < numNeurons; i++){
        tempPtr -> value = 0;
        //if @ output layer 
        if(numNeuronsNextLayer == 0){
            tempPtr -> weightsPtr = NULL;
            tempPtr -> bias = 0;
        }
        else{
            tempPtr -> weightsPtr = malloc(sizeof(float) * numNeuronsNextLayer);
            //initialize the weights 
            float* tempWeightsPtr = tempPtr -> weightsPtr;
            weightInitialization(tempWeightsPtr, numNeuronsNextLayer);
            tempPtr -> bias = 0;
        }
    tempPtr++; // point to the next struct and repeat 
    }

    /* Fill in the paramters of the layer struct */
    neuralLayer* Layer = malloc(sizeof(neuralLayer));
    Layer -> layerPtr = LayerPtr;
    Layer -> numNeurons = numNeurons;
    Layer -> prevLayer = NULL; 
    Layer -> nextLayer = NULL;   
    return Layer;
}

/* A function that will return a pointer to a block of input Neuron structs */
neuralNet* generateNNModel(int numInputNeurons, int numHiddenLayers, int numHiddenNeuronsPerLayer, int numOutputNeurons){
    /* 1. Create the structure of a Neural Network */
    // create a pointer to the (contiguous) input Neuron structs 
    //printf("Neural Network Details:\nNumber of inputs: %d\nNumber of hidden layers: %d\nNumber of Neurons per hidden layer: %d\nNumber of outputs: %d\n\n", numInputNeurons, numHiddenLayers, numHiddenNeuronsPerLayer, numOutputNeurons);
    neuralLayer* inputLayer =  layerInitialization(numInputNeurons, numHiddenNeuronsPerLayer);
    //printf("successfully created the input layer \n");
    neuralLayer* layerIterator = inputLayer;
    for(int i = 0; i < numHiddenLayers; i++){
        if(i == numHiddenLayers - 1){
            neuralLayer* newHiddenLayer = layerInitialization(numHiddenNeuronsPerLayer, numOutputNeurons);
            layerIterator -> nextLayer = newHiddenLayer;
            newHiddenLayer -> prevLayer = layerIterator;
            layerIterator = layerIterator -> nextLayer;
        }
        else{
            neuralLayer* newHiddenLayer = layerInitialization(numHiddenNeuronsPerLayer, numHiddenNeuronsPerLayer);
            layerIterator -> nextLayer = newHiddenLayer;
            newHiddenLayer -> prevLayer = layerIterator;
            layerIterator = layerIterator -> nextLayer;
        }
        
    }
    neuralLayer* outputLayer = layerInitialization(numOutputNeurons, 0);
    layerIterator -> nextLayer = outputLayer;
    outputLayer -> prevLayer = layerIterator;

    neuralNet* model = malloc(sizeof(neuralNet));
    model -> inputLayer = inputLayer; 
    model -> numInputNeurons = numInputNeurons;
    model -> numHiddenLayers = numHiddenLayers;
    model -> numNeuronsPerHiddenLayer = numHiddenNeuronsPerLayer;
    model -> numOutputNeurons = numOutputNeurons;
    return model;
}

void printWeights(float* weightsMemoryBlock, int numWeights){
    for(int i = 0; i < numWeights; i++){
        printf("w%d:%5.2f | ", i, *weightsMemoryBlock);
        weightsMemoryBlock++;
    }
    printf("\n");
}

void modelInfo(neuralNet* model){
    /* Print a summary of a neural network model */
    printf("Neural Network Details:\nNumber of inputs: %d\nNumber of hidden layers: %d\nNumber of Neurons per hidden layer: %d\nNumber of outputs: %d\n\n", model -> numInputNeurons, model -> numHiddenLayers, model -> numNeuronsPerHiddenLayer, model -> numOutputNeurons);

    /* Print the details pertaining to the input layer*/
    neuralLayer* layerPointer = model -> inputLayer;
    Neuron* neuronPtr = layerPointer -> layerPtr;
    printf("Input layer details:\n");
    for(int i = 0; i < model -> numInputNeurons; i++){
        printf("Input Neuron %d: value: %5.2f | bias: %5.2f | ", i, neuronPtr -> value, neuronPtr -> bias);
        printWeights(neuronPtr -> weightsPtr, model -> numNeuronsPerHiddenLayer);
        neuronPtr++;
    }
    printf("\n");

    /* Print the details pertaining to the hidden nodes */
    layerPointer = layerPointer -> nextLayer; //point to the start of the first hidden layer
    for(int i = 0; i < model -> numHiddenLayers; i++){
        printf("Hidden layer %d details:\n", i);
        neuronPtr = layerPointer -> layerPtr;
        neuralLayer* nextLayerPtr = layerPointer -> nextLayer;
        for(int j = 0; j < layerPointer -> numNeurons; j++){
            printf("Hidden Neuron %d: value: %5.2f | bias: %5.2f | ", j, neuronPtr -> value, neuronPtr -> bias);
            printWeights(neuronPtr -> weightsPtr, (nextLayerPtr -> numNeurons));
            neuronPtr++;
        }
        printf("\n");
        layerPointer = layerPointer -> nextLayer;
    }

    /* print out the values of the output neurons */
    neuronPtr = layerPointer -> layerPtr;
    for(int i = 0; i < layerPointer -> numNeurons; i++){
        printf("Output neuron %d value: %5.2f\n", i, neuronPtr -> value);
        neuronPtr++;
    }
    printf("\n");
}

/* A function that performs a feedforward operation, returns 1 if the operation can be performed again 0 otherwise */
void feedForwardHelper(neuralLayer* currentLayer){
    Neuron* currentNeuron = currentLayer -> layerPtr;
    Neuron* nextLayerNeuron = currentLayer -> nextLayer -> layerPtr;
    for(int i = 0; i < currentLayer -> numNeurons; i++){
        float* weightPtr = currentNeuron -> weightsPtr;
        for(int j = 0; j < currentLayer -> nextLayer -> numNeurons; j++){
            nextLayerNeuron[j].value = weightPtr[j] * currentNeuron[i].value;
        }
    }
    
    for(int k = 0; k < currentLayer -> nextLayer -> numNeurons; k++){
        nextLayerNeuron[k].value = sigmoid(nextLayerNeuron[k].value + nextLayerNeuron[k].bias);
    }
}

void feedForward(neuralNet* model){
    neuralLayer* currentLayer = model -> inputLayer;
    for(int i = 0; i < model -> numHiddenLayers + 1; i++){
        feedForwardHelper(currentLayer);
        currentLayer = currentLayer -> nextLayer;
    }
}

int readInputValues(char arr[], neuralNet* model){
    FILE* fpointer;
    fpointer = fopen(arr, "r");
    char line[10];
    
    Neuron* currentNeuron = model -> inputLayer ->layerPtr;

    int i = 0;
    while(fgets(line, 10, fpointer) != NULL){
        puts(line);
        currentNeuron[i].value = atof(line); 
        i++;
    }

    fclose(fpointer);
    return i;
}



int main(int argc, char* argv[]){
    /* initialize the rand functions */
    srand((long)time(NULL));
    
    /* Generate a model */
    double timeElapsed = 0.0;
    clock_t start = clock();
    neuralNet* testModel = generateNNModel(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
    clock_t end = clock();
    
    /* Load input values for the neural network */
    int numValsRead = readInputValues(argv[1], testModel); 
    if(numValsRead != testModel -> numInputNeurons){
        printf("!!!WARNING: ERROR IN READING IN VALUES!!!");
    }

    /*run feed forward*/
    feedForward(testModel);
    
    timeElapsed += (double)(end - start)/CLOCKS_PER_SEC;
    printf("Time elapsed is %f seconds \n", timeElapsed);

    /*check changes*/
    if(atoi(argv[6]) == 1){
        modelInfo(testModel);
    }

       
    return 0;
}

