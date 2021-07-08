import os
import numpy as np
from matplotlib import pyplot as plt

def readCharacterFiles():

    path = 'characters-all/'
    fileNames = os.listdir(path)
    fileNames.sort()
    
    characters = []
    for fileName in fileNames:
        f = open(path + fileName, 'r')
        readen = f.read()
        array = np.array(readen.split(','))
        array2 = np.array([0] * len(array)) # you can use dtype=np.int8 for less space usage
                                            # but for this project we don't need it

        for i in range(0, len(array)): 
            array2[i] = int(array[i])
        characters.append(array2)
        f.close()
    return characters # returns a list which includes 21 arrays for 7 (#character) * 3 (#font)




def readWeightsFile(num_of_inputs, num_of_outputs, path):
    # non hidden layer:
    num_of_weights = num_of_inputs * num_of_outputs

    weights = np.array([0] * num_of_weights, dtype=np.float32)

    f = open(path, 'r')
    weights_list = f.read().split(',')
    f.close()
    
    if(len(weights_list) == num_of_inputs * num_of_outputs):
        for i in range(0, len(weights_list)):
            weights[i] = float(weights_list[i])
    else:
        for i in range(num_of_weights):
            weights[i] = 0.0
    

    return weights

def updateWeightsFile(weights, updateCount, path):

    f = open(path, 'w')
    for i in range(0, len(weights) - 1):
        f.write(str(weights[i]) + ', ')
    f.write(str(weights[len(weights) - 1]))
    f.close()
    return 0


def predict(inputs, weights, weightsIndex, threshold, bipolar_flag):
    activation = 0.0
    weightsIndex = weightsIndex * len(inputs)
    for i in range(len(inputs)):
        activation += weights[weightsIndex] * inputs[i]
        weightsIndex += 1
    if (activation >= threshold):
        return 1.0
    elif bipolar_flag == True:
        if activation < (-1) * threshold:
            return -1.0
    return 0.0

def accuracy(characters, weights, threshold, bipolar_flag):
    
    num_of_outputs = int(len(weights) / len(characters[0])) # 7 = (7*63) / 63
    
    total_correct_predict = 0

    for i in range(len(characters)):
        for j in range(num_of_outputs): # num of output neurons
            result = predict(characters[i], weights, j, threshold, bipolar_flag)
            if j == (i % num_of_outputs): # means same letter -> 1: correct ** 0, -1: wrong answers
                if result == 1:
                    total_correct_predict += 1
            elif bipolar_flag == True:
                if result != 1:
                    total_correct_predict += 1
            else:
                if result == 0:
                    total_correct_predict +=1
    return total_correct_predict / (num_of_outputs * len(characters))

def trainWeights(characters, weights, numof_epochs, learning_rate, threshold, bipolar_flag):
    
    if bipolar_flag == False:
        for character in characters:
            for i in range(len(character)):
                if character[i] == -1:
                    character[i] = 0


    lenInputs = len(characters[0])
    log_accuracy = [0.0] * numof_epochs
    
    for epoch in range(numof_epochs):
        totalError = 0.0
        for i in range(len(characters)):
            inputs = characters[i]
            for j in range(7):
                prediction = predict(inputs, weights, j, threshold, bipolar_flag)
                if j == (i % 7):
                    error = 1 - prediction
                elif bipolar_flag == True:
                    error = -1 - prediction
                else:
                    error = 0 - prediction
                totalError += abs(error)

                inputIndex = 0
                for k in range(j*lenInputs, (j+1)*lenInputs): # range(0,63) | range(63,63*2)
                    # update weights
                    weights[k] += learning_rate * error * inputs[inputIndex]
                    inputIndex += 1 

                print('font', int(i/7) + 1, 'letter:', i % 7, '- neuron: ', j, '- isActive: ', int(prediction))
            print('epoch:', epoch + 1, 'font:', int(i/7) + 1, 'letter:', i % 3, 'total_error:', totalError)

        log_accuracy[epoch] = accuracy(characters, weights, threshold, bipolar_flag)
        print('epoch:', epoch + 1, 'current accuracy:', log_accuracy[epoch])

        if log_accuracy[epoch] == 1.0: # if we reach 100% success rate we can stop
            count = 0 # resize the log_accuracy list with removing zeros
            for i in range(len(log_accuracy)):
                if log_accuracy[i] != 0.0:
                    count += 1
            print('count:',count)
            log_accuracy_no_zeros = [0.0] * count
            for i in range(count):
                log_accuracy_no_zeros[i] = log_accuracy[i]
            return weights, log_accuracy_no_zeros
              
    return weights, log_accuracy

def hammingDistance(characters, font):
    startIndex = -1
    if font == 0:
        startIndex = 0
    elif font == 1:
        startIndex = 7
    elif font == 2:
        startIndex = 14
    else:
        print('Invaild font!')
        return

    distance = np.zeros((7, 7), dtype=np.int)

    for i in range(startIndex, startIndex + 7):
        for j in range(startIndex, startIndex + 7):
            for k in range(len(characters)):
                if characters[i][k] != characters[j][k]:
                    distance[i - startIndex][j - startIndex] += 1
    return distance

    

#
# Reading input characters
#
characters = readCharacterFiles()
distance = hammingDistance(characters, 1)
print(distance)

#
# You can update weights from a file
#
weights = readWeightsFile(63, 7, 'weights.txt')


#
# You can change the values of trainWeights function to make different outputs
#
weights, totalErrorLog = trainWeights(characters, weights, 50, 0.004, 0.5, True)
updateWeightsFile(weights, 0, 'new_weights.txt')

print('total_error_log: ', totalErrorLog)

#
# Plotting the accuracy over time
#
plt.plot(totalErrorLog)
plt.title('Epoch-Accuracy Relationship')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.show()

exit(0)