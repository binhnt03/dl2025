import math
# More activation function you can add here from scratch then save it
# Rectified Linear Unit
def ReLU(matrix:list[list[int]]) -> list[list[int]]:
    result = [[0 for i in range(len(matrix[0]))] for j in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] > 0:
                result[i][j] = matrix[i][j]
            else:
                result[i][j] = 0
    return result

def ReLU_vector(vector: list):
    result = [0 for j in range(len(vector))]
    for i in range(len(vector)):
        if vector[i] > 0:
            result[i] = vector[i]
        else:
            result[i] = 0
    return result

#Softmax
def softmax(matrix: list[list[int]]) -> list[list[int]]:
    result = [[0 for i in range(len(matrix[0]))] for j in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            result[i][j] = (math.e ** matrix[i][j]) / sum([math.e ** matrix[i][n] for n in 
                                                           range(len(matrix[i]))])
    return result

def softmax_vector(vector:list) -> list:
    result = [0 for _ in range(len(vector))]
    for i in range(len(vector)):
        result[i] = (math.e ** vector[i]) / sum(vector)
    return result

