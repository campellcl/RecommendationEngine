#!/usr/bin/env python3

import numpy as np

users = []
ratings = []
movies = []
num_ratings_per_user = [0] * 239127
sum_of_user_ratings = [0] * 239127
average_rating_user = [0] * 239127

def train(users,ratings,movies):
    data = np.load('test.npy')
    idx = np.lexsort(data[:, 1::-1].T)
    data = data[idx, :]
    for rating in data:
        sum_of_user_ratings[rating[0]] += rating[2]
        num_ratings_per_user[rating[0]] += 1
        users.append(rating[0])
        movies.append(rating[1])
        ratings.append(rating[2])
    get_average()
    print (average_rating_user)

def get_average ():
    for i in range(0,239127):
        try:
            average_rating_user[i] = sum_of_user_ratings[i] / num_ratings_per_user[i]
        except ZeroDivisionError as error:
            pass      # set the value to 0
     
def main():
    train(users,movies,ratings)
     
if __name__ == "__main__":
    main()
