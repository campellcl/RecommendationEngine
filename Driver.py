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
    num_movies = np.max(data[:, 1], axis=0)
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
    num_movies = np.max(data[:, 1], axis=0) + 1
    print('%d movies' % num_movies)
    #The statement below initializes an np column vector with all zero's
    h = np.zeros((num_movies, 1))

    '''
    What is the below code doing?
    '''
    k0 = 0
    for j in range(num_movies):
        #print('%5.1f%%' % (100 * j / num_movies), end='\r')
        sys.stdout.write('\rLoading:%5.1f%%' % (100 * j / num_movies))
        #sys.stdout.flush()
        k1 = k0 + 1
        while k1 < len(data) and data[k1, 1] == j:
            k1 += 1
        h[j] = np.mean(data[k0:k1, 2])
        k0 = k1

    #for j in range(num_movies):
    #    index = data[:, 1] == j
    #    h[j] = len(data[index, :])

    #for row in data:
    #    j = row[1]
    #    h[j] += 1

    #j = np.argmin(h, axis=0)
    #print('%d ratings for movie %d' % (h[j], j))
    end_time = time.time()
    print('\nTotal Runtime: %f seconds' % (end_time - start_time))

    '''
    # Sort Data by movie then user (j then i)
    index = np.lexsort(data[:, :2].T)
    data = data[index, :]
    '''

    '''
    # Sort Data by user then movie (i then j)
    index = np.lexsort(data[:, 1::-1].T)
    data = data[index, :]
    '''

    '''
    Start SVD Netflix Recommendation System Code:
    '''
    i = 0
    for user in data[:, 0]:
        for rating in data[i]:
            pass
        i += 1

if __name__ == '__main__':
    main(sys.argv)
