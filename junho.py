'''
n
w1 w2 w3 w4 ... w_{k1}
b1 b2 b3 b4 ... b_{k2}
...

w1 w2 w3 w4 ... w_{k_{n}}
b1 b2 b3 b4 ... b_{k_{n}}

'''


def load_model(filename):
    f = open(filename, "r")

    depth = int(f.readline())

    matrix = []

    for i in range(depth):

        weight = map(int, f.readlines().split())
        bias = map(int, f.readlines().split())

        matrix_elem = [weight, bias]

        matrix.append(matrix_elem)

    return matrix
