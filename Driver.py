"""
    Driver.py
        Driver program for the Netflix Recommendation Engine Project. Takes as input the following:
        :param cmd_args -tdf
        1. The name of the .npy file containing data for training.
        2. The name of the .npy file containing test data for cross validation.
        3. The name of the .npy file containing the instructor provided validation data.
    @Author: Chris Campell
    @Date: 4/11/2016
    @Version: 2.0
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
    start_time = time.time()
    data = np.load(cmd_args[1])
    test = np.load(cmd_args[2])
    validate = np.load(cmd_args[3])

    num_ratings = len(data)
    # np.unique() Returns the sorted unique elements of the movie vector.
    num_movies = len(np.unique(data[:, 1]))
    # ? = np.max(data[:, 1], axis=0) + 1
    num_users = len(np.unique(data[:, 0]))
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
    print('%d unique movies' % num_movies)

    '''
    Create useful matrices sorted differently for data retrieval.
    '''
    # Sort Data by movie then user (j then i)
    index = np.lexsort(data[:, 0:2].T)
    pv = data[index, :]
    index_validate = np.lexsort(validate[:, 0:1].T)
    pval = validate[index_validate, :]

    # Sort Data by user then movie (i then j)
    index = np.lexsort(data[:, 1::-1].T)
    pu = data[index, :]

    '''
    Column Processing Code:
    '''
    pvA = processColumns(pv, num_movies, movie_matrix_mean)
    puA = processColumns(pu, num_users, movie_matrix_mean)
    pval = processValidationColumns(pval, len(pval), movie_matrix_mean)
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
    print("Modeling Linear Equation A: r_{i,j} = m where m = %f and r_{i,j} = %f"
        %(movie_matrix_mean, movie_matrix_mean),flush=True)
    prediction_matrix = np.zeros((len(data), ))
    # prediction_matrix.reshape((-1,)) - data[:,2]
    np.ndarray.fill(prediction_matrix, movie_matrix_mean)
    rmse_model_a = np.sqrt(np.mean((prediction_matrix - data[:, 2]) ** 2))
    print("RMSE of Linear Model A: %f" %rmse_model_a, flush=True)

    # linear equation B: r_{i,j} = m + (a_{i} - m)
    print("Modeling Linear Equation B: r_{i,j} = m + a_{i} where m = %f, and np.mean(a) = %f"
          %(movie_matrix_mean, np.mean(puA)),flush=True)
    rmse_model_b = testModelB(user_weights=puA, movie_matrix_mean=movie_matrix_mean, test=test)
    print("RMSE of Linear Model B: %f" %rmse_model_b, flush=True)

    # linear equation C: r_{i,j} = m + (b_{j} - m)
    print("Modeling Linear Equation C: r_{i,j} = m + b_{j} where m = %f, and np.mean(b) = %f"
          %(movie_matrix_mean, np.mean(pvA)), flush=True)
    rmse_model_c = testModelC(movie_weights=pvA, movie_matrix_mean=movie_matrix_mean, test=test)
    print("RMSE of Linear Model C: %f" %rmse_model_c, flush=True)

    # linear equation D: r_{i,j} = m + (a_{i} - m) + (b_{j} - m)
    print("Modeling Linear Equation D: r_{i,j} = m + a_{i} + b_{j} where m = %f, np.mean(a) = %f, and np.mean(b) = %f"
          %(movie_matrix_mean, np.mean(puA), np.mean(pvA)), flush=True)
    rmse_model_d = testModelD(user_weights=puA, movie_weights=pvA, movie_matrix_mean=movie_matrix_mean, test=test)
    print("RMSE of Linear Model D: %f" %rmse_model_d, flush=True)

    # linear equation C: r_{i,j} = m + (b_{j} - m)
    print("VALIDATE: Modeling Linear Equation C: r_{i,j} = m + b_{j} where m = %f, and np.mean(b) = %f"
          %(movie_matrix_mean, np.mean(pval)), flush=True)
    rmse_validated = testModelC(movie_weights=pvA, movie_matrix_mean=movie_matrix_mean, test=test)
    print("RMSE of VALIDATED Model C: %f" %rmse_validated, flush=True)

    end_time = time.time()
    print('Total Runtime: %f seconds' % (end_time - start_time))

def testModelB(user_weights, movie_matrix_mean, test):
    test_data_length = len(test)
    prediction_matrix = np.zeros((test_data_length, ))
    for j in range(test_data_length):
        sys.stdout.write('\rTesting Model B: %5.1f%%' % (100 * j / test_data_length))
        sys.stdout.flush()
        user_id = test[j, 0]
        prediction_matrix[j] = user_weights[user_id] + movie_matrix_mean
    sys.stdout.flush()
    print("\n")
    return np.sqrt(np.mean((prediction_matrix[:] - test[:, 2]) ** 2))

def testModelC(movie_weights, movie_matrix_mean, test):
    test_data_length = len(test)
    prediction_matrix = np.zeros((test_data_length, ))
    for j in range(test_data_length):
        sys.stdout.write('\rTesting Model C: %5.1f%%' % (100 * j / test_data_length))
        sys.stdout.flush()
        movie_id = test[j, 1]
        prediction_matrix[j] = movie_weights[movie_id] + movie_matrix_mean
    print("\n")
    return np.sqrt(np.mean((prediction_matrix[:] - test[:, 2]) ** 2))

def testModelD(user_weights, movie_weights, movie_matrix_mean, test):
    test_data_length = len(test)
    prediction_matrix = np.zeros((test_data_length, ))
    for j in range(test_data_length):
        sys.stdout.write('\rTesting Model D: %5.1f%%' % (100 * j / test_data_length))
        sys.stdout.flush()
        user_id = test[j, 0]
        movie_id = test[j, 1]
        prediction_matrix[j] = user_weights[user_id] + movie_weights[movie_id] + movie_matrix_mean
    print("\n")
    return np.sqrt(np.mean((prediction_matrix[:] - test[:, 2]) ** 2))

def testModelBAdvanced(prediction_matrix, test):
    length_prediction = len(prediction_matrix)
    rmse_user_matrix = np.zeros((length_prediction, ))
    #predicted base index:
    p0 = 0
    #actual base index:
    a0 = 0
    for j in range(length_prediction):
        sys.stdout.write('\rTesting: %5.1f%%' % (100 * j / length_prediction))
        sys.stdout.flush()
        a1 = a0 + 1
        while a1 < len(test) and test[a1, 0] == j:
            a1 += 1
        if (test[:, 0][j] != j):
            # The user is not in the test data. RMSE for user not possible.
            rmse_user_matrix[j] = np.NaN
        else:
            mean_actual_user_rating = np.mean(test[a0:a1, 2])
            rmse_user_matrix[j] = np.sqrt(np.mean((prediction_matrix[j] - mean_actual_user_rating) ** 2))
            a0 = a1
    return rmse_user_matrix

def validationModelC(movie_weights, movie_matrix_mean, test):
    test_data_length = len(test)
    prediction_matrix = np.zeros((test_data_length, ))
    for j in range(test_data_length):
        sys.stdout.write('\rTesting VALIDATION Model C: %5.1f%%' % (100 * j / test_data_length))
        sys.stdout.flush()
        movie_id = test[j, 1]
        prediction_matrix[j] = movie_weights[movie_id] + movie_matrix_mean
    with open('results.txt', 'w') as fp:
        lines = ['%f\n' % x for x in prediction_matrix]
        fp.writelines(lines)
    print("\n")
    return np.sqrt(np.mean((prediction_matrix[:] - test[:, 2]) ** 2))

def processValidationColumns(data, length, mean):
    h = np.zeros((length,))
    k0 = 0
    for j in range(length):
        sys.stdout.write('\rTraining VALIDATION data: %5.1f%%' % (100 * j / length))
        sys.stdout.flush()
        k1 = k0 + 1
        while k1 < len(data) and data[k1, 1] == j:
            k1 += 1
        h[j] = np.mean(data[k0:k1, 1]) - mean
        k0 = k1
    sys.stdout.flush()
    print("\n")
    return h

def processColumns(data, length, mean):
    h = np.zeros((length,))
    k0 = 0
    for j in range(length):
        sys.stdout.write('\rTraining: %5.1f%%' % (100 * j / length))
        sys.stdout.flush()
        k1 = k0 + 1
        while k1 < len(data) and data[k1, 1] == j:
            k1 += 1
        h[j] = np.mean(data[k0:k1, 2]) - mean
        k0 = k1
    print("\n")
    return h

def testAverage(data, puA, pvA, mean):
    print("running tests")
    h = np.zeros((len(data),))
    index = 0
    for x in data:
        h[index] = (puA[x[0]] + pvA[x[1]] + mean - x[2])
        index += 1
    return h

if __name__ == '__main__':
    main(sys.argv)
