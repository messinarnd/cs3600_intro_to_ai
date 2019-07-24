import NeuralNetUtil
import NeuralNet
import Testing


print " ~ Pen Data! ~ "

penList = []

for i in range(0,5):
	print "Iteration #" + str(i + 1)
	penTest = Testing.testPenData()
	neuralNet = penTest[0]
	testAccuracy = penTest[1]
	print penTest
	penList.append(testAccuracy)
	i = i + 1

print "You have finished 5 iterations!"
print "Average of the Accuracy:" + str(Testing.average(penList))
print "Standard Deviation of the Accuracy:" + str(Testing.stDeviation(penList))
print "Maximum Accuracy:" + str(max(penList))


print " ~ Car Data! ~ "

carList = []

for i in range(0,5):
	print "Iteration #" + str(i + 1)
	carTest = Testing.testCarData()
	neuralNet = carTest[0]
	testAccuracy = carTest[1]
	print carTest
	carList.append(testAccuracy)
	i = i + 1

print "You have finished 5 iterations!"
print "Average of the Accuracy:" + str(Testing.average(carList))
print "Standard Deviation of the Accuracy:" + str(Testing.stDeviation(carList))
print "Maximum Accuracy:" + str(max(carList))




