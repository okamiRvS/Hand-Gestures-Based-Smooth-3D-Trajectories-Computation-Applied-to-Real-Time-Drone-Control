import numpy as np
import pdb

class queueObj():

    def __init__(self, lenMaxQueue):
        self.lenMaxQueue = lenMaxQueue
        self.indexQueue = 0
        self.queue = np.zeros((lenMaxQueue,2), dtype=np.int32)

    def isFullQueue(self):
        if self.indexQueue < self.lenMaxQueue - 1:
            return False
        else:
            return True

    def add(self, val):
        self.queue[self.indexQueue] = val
        
        if self.isFullQueue():
            self.indexQueue = 0
        else:
            self.indexQueue += 1
    
    def get(self, val, idx):
        return self.queue[idx]


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