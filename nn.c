/* 
Author: Sujoy Purkayastha 
random function generator adopted from: http://www.cs.utsa.edu/~wagner/CS2073/random/random.html
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


/* A struct that represents the Neurons within a neural network */
typedef struct{ 
    float value;                  // the value of a Neuron struct  
    float* weightsPtr;            // a pointer to a float array of weights corresponding to each next neuron 
    float bias;                   // the bias of a node 
    //struct Neuron* prevNeurons; // a pointer to a malloc-ed region of memory with the prev Neurons
    //struct Neuron* nextNeurons; // a pointer to a malloc-ed region of memory with the next Neurons
} Neuron;

/* A struct that contains a pointer to a contiguous block of Neuron structs in memory */
/* Will be used to make interfacing layers of Neurons easier */
typedef struct{
    Neuron* layerPtr;
    int numNeurons;
    struct neuralLayer* prevLayer;
    struct neuralLayer* nextLayer;
} neuralLayer; 

/* A struct that encapsulates an entire Neural Network model */
typedef struct {
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
    Neuron* LayerPtr = (Neuron*)malloc(sizeof(Neuron) * numNeurons);
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
            tempPtr -> weightsPtr = (float*)malloc(sizeof(float) * numNeuronsNextLayer);
            //initialize the weights 
            float* tempWeightsPtr = tempPtr -> weightsPtr;
            weightInitialization(tempWeightsPtr, numNeuronsNextLayer);
            tempPtr -> bias = 0;
        }
    tempPtr++; // point to the next struct and repeat 
    }

    /* Fill in the paramters of the layer struct */
    neuralLayer* Layer = (neuralLayer*)malloc(sizeof(neuralLayer));
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

    neuralNet* model = (neuralNet*)malloc(sizeof(neuralNet));
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
        printf("Input Neuron %d: value: %f | bias: %f | ", i, neuronPtr -> value, neuronPtr -> bias);
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
            printf("Hidden Neuron %d: value: %f | bias: %f | ", j, neuronPtr -> value, neuronPtr -> bias);
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
int test;

/* A function that performs a feedforward operation, returns 1 if the operation can be performed again 0 otherwise */
int feedForward(neuralLayer* currentLayer){
 
}


int main(){
    srand((long)time(NULL));

    /* Print the size of the Neuron struct */
    printf("The size of a Neuron is %ld bytes. \n", sizeof(Neuron));
    
    /* Testing the sigmoid function */
    for(int i = -10; i < 11; i++){
        printf("sigmoid(%d) = %f\n", i, sigmoid((float)i));
    }

    /* Test the random number generator */
    printf("test %f \n\n" , randDouble2(-1, 1));

    /* Generate a model */
    neuralNet* testModel = generateNNModel(10, 5, 5, 2);

    /* Print a model summary */
    modelInfo(testModel);
}

