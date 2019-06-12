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
} Neuron;

/* A struct that contains a pointer to a contiguous block of Neuron structs in memory */
/* Will be used to make interfacing layers of Neurons easier */
typedef struct neuralLayer{
    Neuron* layerPtr;
    int numNeurons;
    float bias;                   // the bias of a node 
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

/* Returns the derivative of the sigmoid function at a point x */
/* Assumes that x is the output of the normal sigmoid function */
float dSigmoid(float x){
    return x * (1 - x);
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
        //*iterator = .1;
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
        }
        else{
            tempPtr -> weightsPtr = malloc(sizeof(float) * numNeuronsNextLayer);
            //initialize the weights 
            float* tempWeightsPtr = tempPtr -> weightsPtr;
            weightInitialization(tempWeightsPtr, numNeuronsNextLayer);
        }
    tempPtr++; // point to the next struct and repeat 
    }

    /* Fill in the paramters of the layer struct */
    neuralLayer* Layer = malloc(sizeof(neuralLayer));
    Layer -> layerPtr = LayerPtr;
    Layer -> numNeurons = numNeurons;
    Layer -> bias = .2;
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
        printf("Input Neuron %d: value: %7.4f | ", i, neuronPtr -> value);
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
            printf("Hidden Neuron %d: value: %7.4f | ", j, neuronPtr -> value);
            printWeights(neuronPtr -> weightsPtr, (nextLayerPtr -> numNeurons));
            neuronPtr++;
        }
        printf("Layer bias: %7.4f\n", layerPointer -> bias);
        printf("\n");
        layerPointer = layerPointer -> nextLayer;
    }

    /* print out the values of the output neurons */
    neuronPtr = layerPointer -> layerPtr;
    for(int i = 0; i < layerPointer -> numNeurons; i++){
        printf("Output layer details:\n");
        printf("Output neuron %d value: %7.4f\n", i, neuronPtr -> value);
        neuronPtr++;
    }
    printf("Layer bias: %7.4f\n", layerPointer -> bias);
    printf("\n");
}

/* A function that performs a feedforward operation, returns 1 if the operation can be performed again 0 otherwise */
void feedForwardHelper(neuralLayer* currentLayer){
    Neuron* currentNeuron = currentLayer -> layerPtr;
    Neuron* nextLayerNeuron = currentLayer -> nextLayer -> layerPtr;
    for(int i = 0; i < currentLayer -> numNeurons; i++){
        float* weightPtr = currentNeuron[i].weightsPtr;
        for(int j = 0; j < currentLayer -> nextLayer -> numNeurons; j++){
            nextLayerNeuron[j].value += weightPtr[j] * currentNeuron[i].value;
        }
    }

    for(int k = 0; k < currentLayer -> nextLayer -> numNeurons; k++){
        nextLayerNeuron[k].value = sigmoid(nextLayerNeuron[k].value + (currentLayer -> nextLayer -> bias));
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

float* errorArray(neuralNet* model, float* testValues){
    float* errors = malloc(sizeof(float) * model -> numOutputNeurons + 1);
    neuralLayer* layerPtr = model -> inputLayer;
    for(int i = 0; i < model -> numHiddenLayers + 1; i++){
        layerPtr = layerPtr -> nextLayer; 
    }
    Neuron* NeuronPtr = layerPtr -> layerPtr;
    float totalError = 0;
    for(int i = 0; i < model -> numOutputNeurons; i++){
        totalError += .5 * pow(NeuronPtr[i].value - testValues[i], 2);
        errors[i] = .5 * pow(NeuronPtr[i].value - testValues[i], 2); 
    }
    errors[model -> numOutputNeurons] = totalError;
    return errors;
}

float* loadOutputVals(neuralNet* model){
    float* outputs = malloc(sizeof(float) * model -> numOutputNeurons);
    neuralLayer* layerPtr = model -> inputLayer;
    for(int i = 0; i < model -> numHiddenLayers + 1; i++){
        layerPtr = layerPtr -> nextLayer; 
    }
    Neuron* NeuronPtr = layerPtr -> layerPtr;
    for(int i = 0; i < model -> numOutputNeurons; i++){
        outputs[i] =  NeuronPtr[i].value;
    }
    return outputs;
}

float backPropagationHelper(float* weightsPtr, neuralLayer* layerPointer, int index, float* outputs, float* desiredVals){
    float val = 0;
    if(layerPointer -> nextLayer == NULL){
        Neuron* outputNeurons = layerPointer -> layerPtr;
        val = (outputs[index] - desiredVals[index]) * dSigmoid(outputNeurons[index].value) * weightsPtr[index];
        return val;
    }
    else{
        for(int i = 0; i < layerPointer -> numNeurons; i++){
        Neuron* destNeurons = layerPointer -> nextLayer -> layerPtr;
        Neuron* sourceNeurons = layerPointer ->  layerPtr;
        //val += backPropagationHelper(sourceNeurons -> weightsPtr, layerPointer -> nextLayer, i, outputs, desiredVals) * dSigmoid(destNeurons[i].value) * weightsPtr[i];
        val += backPropagationHelper(sourceNeurons[index].weightsPtr, layerPointer -> nextLayer, i, outputs, desiredVals);
        }
    }
    
    return val; 
}

float dWCalc(neuralNet* model, Neuron sourceNeuron, Neuron destNeuron, neuralLayer* destLayer, int layerIndex, int subIndex, float* outputs, float* desiredVals){
    float alpha = 0.01;
    float deltaW = 0;
    //special case if a weight is between a hidden neuron and an output neuron 
    //do not perform recursive backpropagation 
    if(layerIndex == model -> numHiddenLayers){
        deltaW = alpha * ((destNeuron.value) - desiredVals[subIndex]) * dSigmoid(destNeuron.value) * sourceNeuron.value;
    }
    //the destination neuron is in a hidden layer so recursive back propagation must be applied to traverse through 
    else
    {
        deltaW = alpha * backPropagationHelper(sourceNeuron.weightsPtr, destLayer, subIndex, outputs, desiredVals) * dSigmoid(destNeuron.value) * sourceNeuron.value;
    }
    return deltaW;
}

/* the main driver for the backPropagation algorithm */
void backPropagation(neuralNet* model, float* desiredVals){
    // loop through the input and all hidden layers
    neuralLayer* layerPointer = model -> inputLayer;
    float* outputs = loadOutputVals(model);
    for(int i = 0; i < model -> numHiddenLayers + 1; i++){
        //loop through all of the neurons in a particular layer
        Neuron* sourceNeuronPointer = layerPointer -> layerPtr;
        Neuron* destNeuronPointer = layerPointer -> nextLayer -> layerPtr;
        for(int j = 0; j < layerPointer -> numNeurons; j++){
            //loop through all of the weights in a Neuron's weightPtr
            float* neuronWeightPtr = sourceNeuronPointer -> weightsPtr;
            for(int k = 0; k < layerPointer -> nextLayer -> numNeurons; k++){
                //passing on the next layer because it might be the output layer
                float dW = dWCalc(model, sourceNeuronPointer[j], destNeuronPointer[k], layerPointer -> nextLayer, i, k, outputs, desiredVals);
                neuronWeightPtr[k] -= dW;
            }
        }
        layerPointer = layerPointer -> nextLayer;
    }
}

int main(int argc, char* argv[]){
    

    //creating the input neurons
    Neuron* inputNeurons = malloc(sizeof(Neuron) * 3); 
    inputNeurons[0].value = 1;
    inputNeurons[1].value = 4;
    inputNeurons[2].value = 5;
    float* inputWeight1 = malloc(sizeof(float) * 2);
    inputWeight1[0] = .1;
    inputWeight1[1] = .2;
    inputNeurons[0].weightsPtr = inputWeight1;
    float* inputWeight2 = malloc(sizeof(float) * 2);
    inputWeight2[0] = .3;
    inputWeight2[1] = .4;
    inputNeurons[1].weightsPtr = inputWeight2;
    float* inputWeight3 = malloc(sizeof(float) * 2);
    inputWeight3[0] = .5;
    inputWeight3[1] = .6;
    inputNeurons[2].weightsPtr = inputWeight3;
    neuralLayer* inputLayer = malloc(sizeof(neuralLayer));
    inputLayer -> layerPtr = inputNeurons;
    inputLayer -> prevLayer = NULL;
    inputLayer -> nextLayer = NULL;
    inputLayer -> numNeurons = 3;

    Neuron* hiddenNeurons = malloc(sizeof(Neuron) * 2);
    hiddenNeurons[0].value = 0;
    hiddenNeurons[1].value = 0;
    float* hiddenWeight1 = malloc(sizeof(float) * 2);
    hiddenWeight1[0] = .7;
    hiddenWeight1[1] = .8;;
    hiddenNeurons[0].weightsPtr = hiddenWeight1;
    float* hiddenWeight2 = malloc(sizeof(float) * 2);
    hiddenWeight2[0] = .9;
    hiddenWeight2[1] = .1;;
    hiddenNeurons[1].weightsPtr = hiddenWeight2;
    neuralLayer* hiddenLayer = malloc(sizeof(neuralLayer));
    hiddenLayer -> layerPtr = hiddenNeurons;
    hiddenLayer -> nextLayer = NULL;
    hiddenLayer -> prevLayer = inputLayer;
    hiddenLayer -> numNeurons = 2;
    inputLayer -> nextLayer = hiddenLayer;
    hiddenLayer -> bias = .5;

    Neuron* outputNeurons = malloc(sizeof(Neuron) * 2);
    outputNeurons[0].value = 0;
    outputNeurons[1].value = 0;
    neuralLayer* outputLayer = malloc(sizeof(neuralLayer));
    outputLayer -> numNeurons = 2;
    outputLayer -> bias = .5;
    outputLayer -> layerPtr = outputNeurons;
    outputLayer -> prevLayer = hiddenLayer;
    hiddenLayer -> nextLayer = outputLayer;
    outputLayer -> nextLayer = NULL;


    neuralNet* testModel = malloc(sizeof(neuralNet));
    testModel -> inputLayer = inputLayer;
    testModel -> numHiddenLayers = 1;
    testModel -> numNeuronsPerHiddenLayer = 2;
    testModel -> numInputNeurons = 3;
    testModel -> numOutputNeurons = 2;

    modelInfo(testModel);
    feedForward(testModel);
    modelInfo(testModel);
    
    float* desiredVals = malloc(sizeof(float) * 2);
    desiredVals[0] = .1;
    desiredVals[1] = .05;
    backPropagation(testModel, desiredVals);
    modelInfo(testModel);

    return 0;
}

 