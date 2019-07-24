from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData,buildExamplesFromXorData
from NeuralNet import buildNeuralNet
import cPickle 
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData() 
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData,maxItr = 200, hiddenLayerList =  hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData,maxItr = 200,hiddenLayerList =  hiddenLayers)



xorData = buildExamplesFromXorData()
def testXorData(hiddenLayers = [0]):


	return buildNeuralNet(xorData,maxItr = 1600,hiddenLayerList =  hiddenLayers)

	# Resulted in 4 Perceptrons
    #return buildNeuralNet(xorData,maxItr = 200,hiddenLayerList =  hiddenLayers)

    # Resulted in 3 Perceptrons
    #return buildNeuralNet(xorData,maxItr = 500,hiddenLayerList =  hiddenLayers)
    #return buildNeuralNet(xorData,maxItr = 800,hiddenLayerList =  hiddenLayers)
    #return buildNeuralNet(xorData,maxItr = 1100,hiddenLayerList =  hiddenLayers)

    # Resulted in 2 Perceptrons
	#return buildNeuralNet(xorData,maxItr = 1400,hiddenLayerList =  hiddenLayers)
	#return buildNeuralNet(xorData,maxItr = 1700,hiddenLayerList =  hiddenLayers)
	#return buildNeuralNet(xorData,maxItr = 2000,hiddenLayerList =  hiddenLayers)

	# I got tired of going by intervals of 300. I am going to go by 1000's now to see
	# more change

	# Resulted in 2 Perceptrons
	# return buildNeuralNet(xorData,maxItr = 3000,hiddenLayerList =  hiddenLayers)

	# Let's try going by 2,000's?

	# Resulted in 2 Perceptrons
	#return buildNeuralNet(xorData,maxItr = 5000,hiddenLayerList =  hiddenLayers)


