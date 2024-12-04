import numpy as np
def matrix_operation(array1, array2, operation):
    def elementwise_multiplication(arr1, arr2):
        if len(arr1) != len(arr2) or len(arr1[0]) != len(arr2[0]):
            return "Error: Arrays must have the same shape for element-wise multiplication."
        result = [[arr1[i][j] * arr2[i][j] for j in range(len(arr1[0]))] for i in range(len(arr1))]
        return result

    def matrix_multiplication(arr1, arr2):
        if len(arr1[0]) != len(arr2):
            return "Error: Number of columns in array1 must equal the number of rows in array2 for matrix multiplication."
        result = [[sum(arr1[i][k] * arr2[k][j] for k in range(len(arr2))) for j in range(len(arr2[0]))] for i in range(len(arr1))]
        return result

    def determinant(matrix):
        if len(matrix) != len(matrix[0]):
            return "Error: Determinant cannot be computed for non-square matrices."
        
        # Recursive function to compute determinant
        def compute_det(mat):
            if len(mat) == 1:  # Base case for 1x1 matrix
                return mat[0][0]
            if len(mat) == 2:  # Base case for 2x2 matrix
                return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
            det = 0
            for c in range(len(mat)):
                minor = [[mat[i][j] for j in range(len(mat)) if j != c] for i in range(1, len(mat))]
                det += ((-1) ** c) * mat[0][c] * compute_det(minor)
            return det
        
        return compute_det(matrix)

    if operation == "dot":
        return elementwise_multiplication(array1, array2)
    elif operation == "matrix":
        return matrix_multiplication(array1, array2)
    elif operation == "determinant":
        det1 = determinant(array1)
        det2 = determinant(array2)
        return (det1, det2)
    else:
        return "Error: Invalid operation. Choose from 'dot', 'matrix', or 'determinant'."
def take_matrix_input(name):
    rows = int(input(f"Enter the number of rows in {name}: "))
    cols = int(input(f"Enter the number of columns in {name}: "))
    print(f"Enter the elements of {name} row by row (space-separated):")
    matrix = []
    for _ in range(rows):
        row = list(map(int, input().split()))
        if len(row) != cols:
            raise ValueError("Invalid input: Number of elements in the row doesn't match the number of columns.")
        matrix.append(row)
    return matrix
print("Matrix Operations: Choose from 'dot', 'matrix', or 'determinant'")
operation = input("Enter the operation: ").strip()

print("Enter the first matrix:")
array1 = take_matrix_input("Matrix 1")

print("Enter the second matrix:")
array2 = take_matrix_input("Matrix 2")

result = matrix_operation(array1, array2, operation)
print("Result:")
print(result)
