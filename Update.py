import numpy as np
import sys
import time

def main():
    data = np.load("train_small_10.npy")
    num_ratings = len(data)
    num_users = len(np.unique(data[:, 0]))
    num_ratings_per_user = [0] * num_users
    sum_of_user_ratings = [0] * num_users
    average_rating_user = [0] * num_users
    
    #filename = input("Input the filename: ")
    num_movies = len(np.unique(data[:, 1]))
    movie_matrix_mean = np.mean(data[:, 2])
        
    #Sort Data by movie then user (j then i)
    index = np.lexsort(data[:, 0:2].T)
    pv = data[index, :]
    
    # Create the two variables for data to be tested on
    h = processMovies (pv, num_movies)
    
    # Sort data by user then movie (i then j)
    idx = np.lexsort(data[:, 1::-1].T)
    data = data[idx, :]
    
    # Find the average rating per user
    for rating in data:
        sum_of_user_ratings[rating[0]] += rating[2]
        num_ratings_per_user[rating[0]] += 1
    for i in range(0,num_users):
        try:
            average_rating_user[i] = sum_of_user_ratings[i] / num_ratings_per_user[i]
        except ZeroDivisionError as error:
            pass      # set the value to 0

    # Calculate the RMSE for both the users and movies
    # average ratings arrays
    
    RMSE_A = np.sqrt(np.mean((h - data[:, 2]) ** 2))
    print (RMSE_A)

def processMovies(data, length):
    h = np.zeros((length, 1))
    k0 = 0
    for j in range(length):
        sys.stdout.write('\rProcessing Column Request:%5.2f%%' % (100 * j / length))
        k1 = k0 + 1
        while k1 < len(data) and data[k1, 1] == j:
            k1 += 1
        h[j] = np.mean(data[k0:k1, 2])
        k0 = k1
    print("\n")
    return h

if __name__ == "__main__":
    main()
