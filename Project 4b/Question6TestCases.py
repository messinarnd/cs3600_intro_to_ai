# Tanya Churaman

import NeuralNetUtil
import NeuralNet
import Testing


# print " ~ Pen Data! ~ "


# for i in range(0,41,5):
# 	print " ~~ This is a run with " + str(i) + " neurons per hidden layer ~~"

# 	penList = []

# 	for j in range(0,5):
# 		print "Iteration #" + str(j + 1)
# 		penTest = Testing.buildNeuralNet(Testing.penData,maxItr = 200, hiddenLayerList = [i])
# 		neuralNet = penTest[0]
# 		testAccuracy = penTest[1]
# 		print penTest
# 		penList.append(testAccuracy)
# 		j = j + 1

# 	print "You have finished 5 iterations!"
# 	print "Average of the Accuracy:" + str(Testing.average(penList))
# 	print "Standard Deviation of the Accuracy:" + str(Testing.stDeviation(penList))
# 	print "Maximum Accuracy:" + str(max(penList))


print " ~ Car Data! ~ "



for i in range(0,41,5):
	print " ~~ This is a run with " + str(i) + " neurons per hidden layer ~~"

	carList = []

	for j in range(0,5):
		print "Iteration #" + str(j + 1)
		carTest = Testing.buildNeuralNet(Testing.carData,maxItr = 200, hiddenLayerList = [i])
		neuralNet = carTest[0]
		testAccuracy = carTest[1]
		print carTest
		carList.append(testAccuracy)
		j = j + 1

	print "You have finished 5 iterations!"
	print "Average of the Accuracy:" + str(Testing.average(carList))
	print "Standard Deviation of the Accuracy:" + str(Testing.stDeviation(carList))
	print "Maximum Accuracy:" + str(max(carList))




