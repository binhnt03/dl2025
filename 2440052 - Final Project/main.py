from LCG import LinearCongruentialGenerator as lcg
import matrix as mat
import activation as act
import math

random = lcg()
class ConvNN:
    def __init__(self,num_classes:int,input:list[list[float]],filter_size:int, filter:list[list[float]], numconv:int) -> None:
        self.num_classes = num_classes
        self.input = input
        self.filter = filter
        self.out = []
        self.filter_size = filter_size
        self.input_size = len(input)
        self.nconv = numconv
        # Adjust based on the convolved size.
        self.weights = [[random.random() for _ in range((len(input) - numconv*(filter_size-1))**2)]]
        self.biases = [[random.random() for _ in range((len(input) - numconv*(filter_size-1))**2)]]

    def conv_forward(self):
        self.out = mat.conv2Dsq(self.input,self.filter)
        self.input = self.out
        return self.out
    def max_pooling(self,pooling_size:int ,stride:int = 1) -> list[list[float]]:
        rows = len(self.out)
        cols = len(self.out[0])

        row_pool = rows // stride
        col_pool = cols // stride

        result = [[0 for _ in range(col_pool)] for _ in range(row_pool)]

        for i in range(row_pool):
            for j in range(col_pool):
                pool = [self.out[i*stride:i*stride+pooling_size] + self.out[j*stride:j*stride+pooling_size]]
                result[i][j] = max(pool)
                self.out
        return self.out
    def flatten(self) -> list:
        result = []
        for i in range(len(self.out)):
            for j in range(len(self.out[i])):
                result.append(self.out[i][j])
        self.out = result
        return self.out
    def linear_forward(self):
        result = [0 for _ in range(len(self.out))]
        result = mat.add(mat.multiply(self.weights,self.out),self.biases)
        self.out = result
        return self.out
    def ReLU(self):
        self.out = act.ReLU_vector(self.out)
        return self.out
    def Softmax(self):
        self.out = act.softmax_vector(self.out)
        return self.out
    def loss(self, labels):
        # The loss function will be cross-entropy
        loss = -sum(math.log(self.out))
        delta = self.out - labels
        return loss, delta
    def output(self):
        print(self.out)
        return self.out
    def backprop(self, fts, labels:list, delta:list, learning_rate: float):
        self.out = self.out - mat.scalar(mat.scalar(self.out,delta),learning_rate)
    def train(self, num_iter: int, input, true):
        for i in range(self.nconv):
            self.conv_forward(input)
        self.max_pooling(3,2)
        self.linear_forward()
        self.ReLU()
        self.linear_forward()
        self.Softmax()
        loss = self.loss(true)
        for j in range(num_iter):
            self.backprop()

        """ 
        This part is so Hard! I cannot do it! I have been trying but 
        it did not gone well, the backprop is very complicated, from the flattened layer, then convolved layer,
        and finally is the filter!
        """
        
    

if __name__ == "__main__":
    kernel_size = 10
    kernel = [[1 for _ in range(kernel_size)] for _ in range(kernel_size)]
    random_input = [[random.random() for _ in range(28)] for _ in range(28)]
    cnn = ConvNN(2,random_input,kernel_size,kernel,2)
    cnn.conv_forward()
    cnn.conv_forward()
    cnn.max_pooling(pooling_size = 2,stride =2)
    cnn.flatten()
    cnn.output()
    


