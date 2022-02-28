import numpy as np
from numpy.linalg import norm 
import math
np.random.seed(0)
from itertools import combinations

####To get the smallest vector, call shortest_vector(B) function with the basis as an argument to the function.
####The function will return the basis, B, the smallest vector u, the vector x such that u = B*x and the length of the smallest vector.

##checks if vector point on lattice by parity
def is_point_parity(point1, point2):      
    for x in range(0, len(point1)):
        if (point1[x] + point2[x]) % 2 == 0:
            continue
        else:
            return False
    return True 
    
##check if vector point on lattice by ints
def is_point_int(v):
    return (p == math.floor(p) for p in v)
    
##get vector u from x
def get_vector(new_v, B, n):
    u = np.zeros(shape=(1, n))
    for x in range(0, n):
        v = new_v[x] * B[:, x]
        u = np.add(u, v)
    return u[0]

##checks newly found data - appending to dict if needed
##also checks if this vector is minimum found so far
def new_sv(new_v, random_vectors, vectors, u, x, n, B, shortest_vector):

    if np.all(new_v==0):
        return vectors, shortest_vector, u, x

    ##get vector u from B given x
    this_u = get_vector(new_v, B, n)
        
    ##if vector already in list
    for i in vectors:
        if (i[0] == new_v).all():
            return vectors, shortest_vector, u, x

    ##if appending to inital list
    if random_vectors == True:
        vectors.append([new_v, norm(this_u)])

    #sieving - check if new found vector smaller than biggest one in dict
    else:
        norms = [x[1] for x in vectors]
        max_vect = np.argmax(norms)
        ##if this vector smaller than biggest
        ##delete biggest vector and add this one
        if norm(this_u) < vectors[max_vect][1]:
            vectors.pop(max_vect)
            vectors.append([new_v, norm(this_u)])

    ##check if vector is smallest found
    if norm(this_u) < shortest_vector:
        u = this_u
        shortest_vector = norm(this_u)
        x = new_v    
    return vectors, shortest_vector, u, x

##for each vector, find smallest between that one and each other
def sieve(vectors, u, x, shortest_vector, n, B, v1):
    if v1 == False:
        v1 = np.random.randint(-20, 20, size=n)
        v2 = np.random.randint(-20, 20, size=n)
        v2_p = v2 + 1
        v2_n = v2 - 1
        ##if average of v1 and v2 is point
        if is_point_parity(v1, v2):
            new_v = np.add(v2, v2)
            new_v = new_v / 2
            vectors, shortest_vector, u, x = new_sv(new_v, True, vectors, u, x, n, B, shortest_vector)
        ##if average of v1 and v2 + 1 is point
        elif is_point_parity(v1, v2_p):
            new_v = np.add(v1, v2_p)
            new_v = new_v / 2
            vectors, shortest_vector, u, x = new_sv(new_v, True, vectors, u, x, n, B, shortest_vector)
        ##if average of v1 and v2 - 1 is point
        elif is_point_parity(v1, v2_n):
            new_v = np.add(v1, v2_n)
            new_v = new_v / 2
            vectors, shortest_vector, u, x = new_sv(new_v, True, vectors, u, x, n, B, shortest_vector)
        ##check if difference of v1 and v2 is point
        elif is_point_int(np.subtract(v1, v2)):
            new_v = np.subtract(v1, v2)
            vectors, shortest_vector, u, x = new_sv(new_v, True, vectors, u, x, n, B, shortest_vector)
        else:
            return vectors, shortest_vector, u, x
        return vectors, shortest_vector, u, x

    else:
        for vec1, vec2 in combinations(vectors, r = 2):
            v1 = vec1[0]
            v2 = vec2[0]
            ##if same vector
            if (v1 == v2).all():
                continue
            v2_p = v2 + 1
            v2_n = v2 - 1                    
            ##if average of v1 and v2 is point
            if is_point_parity(v1, v2):
                new_v = np.add(v1, v2)
                new_v = new_v / 2
                vectors, shortest_vector, u, x = new_sv(new_v, False, vectors, u, x, n, B, shortest_vector)
            ##if average of v1 and v2 + 1 is point
            elif is_point_parity(v1, v2_p):
                new_v = np.add(v1, v2_p)
                new_v = new_v / 2
                vectors, shortest_vector, u, x = new_sv(new_v, False, vectors, u, x, n, B, shortest_vector)
            ##if average of v1 and v2 - 1 is point
            elif is_point_parity(v1, v2_n):
                new_v = np.add(v1, v2_n)
                new_v = new_v / 2
                vectors, shortest_vector, u, x = new_sv(new_v, False, vectors, u, x, n, B, shortest_vector)
            ##if difference of v1 and v2 is point
            elif is_point_int(np.subtract(v1, v2)):
                new_v = np.subtract(v1, v2)
                vectors, shortest_vector, u, x = new_sv(new_v, False, vectors, u, x, n, B, shortest_vector)
        return vectors, shortest_vector, u, x

##function to call to find the smallest vector
def shortest_vector(B):
    ##basis as numpy array
    B = np.array(B)
    ##number of vectors
    n = len(B[0])
    ##list of lists holding each vector and its length
    vectors = []
    ##smallest vector found
    u = []
    ##x such that u = B*x
    x = []
    ##length of shortest vector
    shortest_vector = np.inf

    ##add inital vectors to list of vectors
    basis_vectors = np.identity(n)
    for i in basis_vectors:
        vectors.append([i, norm(get_vector(i, B, n))])
        if norm(get_vector(i, B, n)) < shortest_vector:
            shortest_vector = norm(get_vector(i, B, n))
            u = get_vector(i, B, n)
            x = i
        
    ##build initial list of vectors
    while len(vectors) < n*7:
        vectors, shortest_vector, u, x = sieve(vectors, u, x, shortest_vector, n, B, False)

    ##sieve through vectors in list, finding smaller ones
    i = 0
    while i < n:
        vectors, shortest_vector, u , x = sieve(vectors, u, x, shortest_vector, n, B, True)
        i += 1
    
    ##make output of u and x the same as example
    final_u = []
    for i in u:
        final_u.append([i])
    final_x = []
    for i in x:
        final_x.append([i])
    
    ##return basis, shortest vector, length of shortest vector and x s.t u = Bx
    return "B = ", B, "u = ", final_u, "norm = ", shortest_vector, "x = ", final_x

B = ([37, 20, 96, 20, 34, 64, 82, 56, 47, 21, 50, 49],
    [39, 24, 19, 49, 82, 97, 88, 84, 41, 51, 36, 74],
    [19, 56, 37, 73,  4, 12, 72, 18, 46,  8, 54, 94],
    [13, 46, 26,  8, 83, 71, 45, 84, 21, 32, 53, 80],
    [65, 39, 25, 56, 52, 44, 84, 30, 69, 33, 13,  5],
    [59, 56, 90,  1, 42, 58, 90, 92,  2,  6,  7, 80],
    [18, 14, 26, 31, 91, 93, 77, 64, 95, 36, 23,  5],
    [11, 58, 22, 51, 90, 13, 93, 43, 21, 81, 12, 77],
    [42, 65, 99,  6, 23, 43, 94, 30, 37, 66, 34, 66],
    [99, 31, 24, 44, 18, 58, 17, 27, 70, 88, 59, 11],
    [30, 43, 21, 70, 48, 47, 13, 93, 94, 48, 69, 58],
    [ 7, 12, 94, 88, 59, 95, 43, 62, 71, 36, 91, 70])

print(shortest_vector(np.array(B)))
