# Multiply tưo matrices
def multiply(matrix_a: list[list[float]],matrix_b: list[list[float]]) -> list[list[float]]:
# Create empty matrix for storing result
    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]
# Perform the matrix multiplication, i for number of rows of matrix A, j for number of columns of matrix B
# -> New size of the new matrix: i x k times k x j = i x j 
# k for the number of rows of Matrix B: this is for performing the dot product.
    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result

# This function is for ONLY SQUARED MATRIX

def conv2Dsq(matrix: list[list[float]],filter: list[list[float]]) -> list[list[float]]:
    # The Convolved matrix size, yes I am doing it by hands for the operation!
    if len(matrix) > len(filter) and len(filter) == len(filter[0]):
        result = [[0 for _ in range(len(matrix) + 1 - len(filter))] for _ in range(len(matrix) + 1 - len(filter))]
    else:
        raise KeyError("The filter size is not appropriate for the matrix, the filter you put in is: "+
                       len(filter)+ " x "+len(filter[0]) +" and the shape of the matrix you put in is: "+
                       len(matrix) + " x " + len(matrix[0]))
    # Let's say that the matrix has size of n x n, filter size of m x m: we will have the convolved size shrunk into 
    # (n-m+1) x (n-m+1) convolved matrix size. And the matrix has a dimension of 2 so, we will do a nested loop, this is 
    # to calculate each entry of the convolved matrix
    # First 2 nested loops is for the adding result to the matrix, next 2 is for convolving
    for i in range(len(result)):
        for k in range(len(result)):
            for m in range(len(filter)):
                for n in range(len(filter)):
                    result[i][k] += matrix[i+m][k+n] * filter[m][n] 

    return result

# Some common matrix operations when needed of course
def add(mata: list[list[float]],matb: list[list[float]]) -> list[list[float]]:
    if len(mata) == len(matb) and len(mata[0]) == len(matb[0]):
        result = [[0 for _ in range(len(mata[0]))] for _ in range(len(mata))]
    else:
        raise KeyError(" The size of the 2 matrices does not match")
    
    for i in range(len(mata)):
        for j in range(len(mata[0])):
            result[i][j] = mata[i][j] + matb[i][j]
    
    return result

def sub(mata: list[list[float]],matb: list[list[float]]) -> list[list[float]]:
    if len(mata) == len(matb) and len(mata[0]) == len(matb[0]):
        result = [[0 for _ in range(len(mata[0]))] for _ in range(len(mata))]
    else:
        raise KeyError(" The size of the 2 matrices does not match")
    
    for i in range(len(mata)):
        for j in range(len(mata[0])):
            result[i][j] = mata[i][j] - matb[i][j]
    
    return result

def scalar(mata: list[list[float]],scale: float) -> list[list[float]]:
    result = [[0 for _ in range(len(mata[0]))] for _ in range(len(mata))]    
    for i in range(len(mata)):
        for j in range(len(mata[0])):
            result[i][j] = mata[i][j] * scale
    
    return result








    

