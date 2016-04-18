"""
    Driver.py
        Driver program for the Netflix Recommendation Engine Project. Takes as input the following:
        1. The name of the .npy file containing data for training.
        2. The name of the .npy file containing test data for cross validation.
        3. TODO: ??
    @Author: Chris Campell, Chris Chalfant, Travis Clark
    @Date: 4/11/2016
    @Version: 1.0
"""
from __future__ import print_function
import numpy as np
import sys
import time


'''
    main -Default main function for driver class.
    @param cmd_args -Default command line arguments, specified above.
'''
def main(cmd_args):
    '''
    The data matrix contains each rating as a row. The first column represents the user index (0-239126),
    the second column is the movie index (0-8884), and the third column is the rating (1-5), with a total of
    20733920 rows.
    '''
    data = np.load(cmd_args[1])
    num_ratings = len(data)
    # np.unique() Returns the sorted unique elements of the movie vector.
    num_movies = len(np.unique(data[:, 1]))
    # ? = np.max(data[:, 1], axis=0) + 1
    num_users = len(np.unique(data[:, 0]))

    start_time = time.time()
    movie_matrix_mean = np.mean(data[:, 2])
    '''
    m = 0
    for row in data:
        m += row[2]
    m /= len(data)
    print(m)
    '''
    end_time = time.time()
    print('Initialization Runtime: %f seconds' % (end_time - start_time))
    print('%f ratings per movie' % (num_ratings / num_movies))
    start_time = time.time()
    #Index = a matrix of just user index and movie index (ratings are trimmed)
    #The np.lexsort() line appears to sort an 'n' dimensional matrix in ascending order after transposition.
    index = np.lexsort(data[:, :2].T)
    #Sort by the movie index? Why are we doing this?
    data = data[index, :]

    #Why are we adding one here? Is this because the index is zero based?
    # num_movies = np.max(data[:, 1], axis=0) + 1
    print('%d movies' % num_movies)

    '''
    Create useful matrices sorted differently for data retrieval.
    '''
    # Sort Data by movie then user (j then i)
    index = np.lexsort(data[:, 0:2].T)
    pv = data[index, :]

    # Sort Data by user then movie (i then j)
    index = np.lexsort(data[:, 1::-1].T)
    pu = data[index, :]

    '''
    Column Processing Code:
    '''
    #The statement below initializes an np column vector with all zero's
    h = processColumns(pv, num_movies)
    l = processColumns(pu, num_users)
    
    #for j in range(num_movies):
    #    index = data[:, 1] == j
    #    h[j] = len(data[index, :])

    #for row in data:
    #    j = row[1]
    #    h[j] += 1

    #j = np.argmin(h, axis=0)
    #print('%d ratings for movie %d' % (h[j], j))

    '''
    Start SVD Netflix Recommendation System Code Validation:
    '''
    # linear equation A: r_{i,j} = m
    print("Modeling Linear Equation: r_{i,j} = m where m = %f and r_{i,j} = %f" %(movie_matrix_mean, movie_matrix_mean))
    # linear equation B: r_{i,j} = m + (a_{i} - m)
    print("Modeling Linear Equation: r_{i,j} = m + a_{i} where m = %f, a_{i} = %f, and r_{i,j} = %f" %(movie_matrix_mean, np.mean(h), (movie_matrix_mean + np.mean(h))))
    # linear equation C: r_{i,j} = m + (b_{j} - m)
    print("Modeling Linear Equation: r_{i,j} = m + b_{j} where m = %f, b_{j} = %f, and r_{i,j} = %f" %(movie_matrix_mean, np.mean(l), (movie_matrix_mean + np.mean(l))))
    # linear equation D: r_{i,j} = m + (a_{i} - m) + (b_{j} - m)
    print("Modeling Linear Equation: r_{i,j} = m + a_{i} + b_{j} where m = %f, a_{i} = %f, b_{j} = %f and r_{i,j} = %f" %(movie_matrix_mean, np.mean(h), np.mean(l), (movie_matrix_mean + np.mean(h) + np.mean(l))))
    prediction_matrix = np.zeros((len(data), ))
    # print("Size of prediction_matrix: %d and Size of num_movies: %d" %(len(prediction_matrix), num_movies))
    # prediction_matrix.reshape((-1,)) - data[:,2]
    np.ndarray.fill(prediction_matrix, movie_matrix_mean)
    rmse_model_a = np.sqrt(np.mean((prediction_matrix - data[:,2]) ** 2))
    print("RMSE Model of Linear Equation A: %f" %rmse_model_a)
    rmse_model_b = None
    rmse_model_c = None
    rmse_model_d = None

    print("RMSE Model B: %f" %rmse_model_b)
    print("RMSE Model C: %f" %rmse_model_c)
    print("RMSE Model D: %f" %rmse_model_d)
    end_time = time.time()
    print('Total Runtime: %f seconds' % (end_time - start_time))
    
def processColumns(data, length):
    h = np.zeros((length, 1))
    k0 = 0
    for j in range(length):
        sys.stdout.write('\rLoading:%5.1f%%' % (100 * j / length))
        k1 = k0 + 1
        while k1 < len(data) and data[k1, 1] == j:
            k1 += 1
        h[j] = np.mean(data[k0:k1, 2])
        k0 = k1
    print("\n") 
    return h
	 
if __name__ == '__main__':
    main(sys.argv)
