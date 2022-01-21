import numpy as np
import pdb
from typing import Tuple


class queueObj():

    def __init__(self, lenMaxQueue: int):

        self.lenMaxQueue = lenMaxQueue
        self.indexQueue = 0
        self.queue = np.zeros((lenMaxQueue,2), dtype=np.int32)

        # This is to check if it's the same gesture, we need to analyze the past
        self.outputClasses = np.zeros((lenMaxQueue), dtype=object)
        self.probabilities = np.zeros((lenMaxQueue), dtype=np.float32)


    def isFullQueue(self) -> bool:
        """
        Return if the the self.queue is full of elements.
        """
        
        if self.indexQueue < self.lenMaxQueue - 1:
            return False
        else:
            return True


    def addMeanAndMatch(self, val: int, match: str, prob: float):
        """
        Function to add the new value during start state (?),
        and add also the name of gesture and its relative probability
        to be that.
        """
        
        self.queue[self.indexQueue] = val

        self.outputClasses[self.indexQueue] = match
        self.probabilities[self.indexQueue] = prob
        
        if self.isFullQueue():
            self.indexQueue = 0
        else:
            self.indexQueue += 1
    

    def get(self, idx: int):
        """
        Return the i-th element of the queue
        """

        return self.queue[idx]


    def checkGesture(self, gesture: str) -> bool:
        """
        Return if the gesture is the last gesture executed. To
        verify we check the queue elements counting how many times
        that gesture appear.
        """

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


    def mean(self) -> Tuple[int,int]:
        """
        Compute the mean of x and y component.
        """

        x_mean, y_mean = self.queue.mean(axis=0)
        return int(x_mean), int(y_mean)
    

    def meanOfTheLastNelements(self, nlastMovements: int):
        """
        Compute the mean of the nlastMovements of x and y component.
        """

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