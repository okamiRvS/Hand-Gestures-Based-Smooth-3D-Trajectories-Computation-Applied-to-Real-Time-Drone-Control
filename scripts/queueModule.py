import numpy as np
import pdb

class queueObj():

    def __init__(self, lenMaxQueue):
        self.lenMaxQueue = lenMaxQueue
        self.indexQueue = 0
        self.queue = np.zeros((lenMaxQueue,2), dtype=np.int32)

        # this is to check if it's the same gesture, we need to analyze the past
        self.outputClasses = np.zeros((lenMaxQueue), dtype=object)
        self.probabilities = np.zeros((lenMaxQueue), dtype=np.float32)

    def isFullQueue(self):
        if self.indexQueue < self.lenMaxQueue - 1:
            return False
        else:
            return True

    def addMeanAndMatch(self, val, match, prob):
        self.queue[self.indexQueue] = val

        self.outputClasses[self.indexQueue] = match
        self.probabilities[self.indexQueue] = prob
        
        if self.isFullQueue():
            self.indexQueue = 0
        else:
            self.indexQueue += 1
    
    def get(self, val, idx):
        return self.queue[idx]

    def checkGesture(self, gesture):
        count = 0

        for i in range(self.lenMaxQueue):
            if self.outputClasses[i] == gesture:
                count+=1 * self.probabilities[i] # we use probabilities as a weight

        # minCount is the 75% of the queue length, a good quantity to be sure that is the right gesture
        minCount = self.lenMaxQueue * 0.75
        
        if count > minCount:
            return True
        else:
            # print("NOTE: could be that in handgesturemodule.py you called the gesture in another way")
            return False

    def mean(self):
        x_mean, y_mean = self.queue.mean(axis=0)
        return int(x_mean), int(y_mean)
    
    def meanOfTheLastNelements(self, nlastMovements):
        tmpList = np.zeros(nlastMovements) # create array with only the last n points inserted into the queue
        if self.indexQueue < nlastMovements:
            nElementFromEnd = nlastMovements - self.indexQueue
            tmpList = self.queue[(self.lenMaxQueue - nElementFromEnd)::]
            tmpList = np.concatenate((tmpList, self.queue[:self.indexQueue]))
        else:
            tmpList = np.array(self.queue[self.indexQueue-nlastMovements:self.indexQueue])
            
        x_mean, y_mean = np.mean(tmpList, axis=0)
        return int(x_mean), int(y_mean)

def main():
   print("hello")

if __name__ == "__main__":
    main()