from django.shortcuts import render
from django.http import JsonResponse
from django.core import serializers
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import os
import json
import pickle
from collections import defaultdict

df = pd.DataFrame()
titles = []
sorted_titles = []
onehot_feat_to_index = {}
onehot_index_to_feat = {}
seen_movies = ["dummy"]
current_user_feat_X = []
current_user_feat_Y = []

def button(request):
    # print(os.getcwd())
    global df, titles, sorted_titles, onehot_feat_to_index, onehot_index_to_feat
    onehot_feat_to_index, onehot_index_to_feat = features_construction()
    if len(titles) > 0:
        movieEntry = fetchFeatures()
        return render(request, 'home.html', {'titles':titles, 'movieData': movieEntry})
    df = pd.read_csv('IMDB_Final_Movies.csv')
    title = df['primaryTitle'].tolist()
    year = df['startYear'].tolist()
    sorted_titles = json.load(open("sorted_movies_for_genres.json"))
    for (i,j) in zip(title,year):
        titles.append(i + '(' + str(int(j)) + ')')
    movieEntry = fetchFeatures()
    return render(request, 'home.html', {'titles': titles, 'movieData': movieEntry})

def fetchFeatures():
    global df, titles, sorted_titles, seen_movies
    longTitle = "dummy"
    while longTitle in seen_movies:
        # longTitle = request.POST.get('title', False)
        gen = random.sample(sorted_titles.keys(),1)[0]
        rand_sample = random.sample(sorted_titles[gen][:100],1)
        longTitle = rand_sample[0][0] + "(" + rand_sample[0][1] + ")"
    title = ""
    if longTitle:
        title = longTitle.split('(')
        ti = title[0]
        # if (df.query('primaryTitle == "%s"' % (ti))).empty:
        #     return render(request, "home.html", {'titles': titles, "titleInvalid":True})
        year = int(title[1].split(')')[0])
    # else:
    #     return render(request, "home.html", {'titles': titles, "titleInvalid":True})
        # Search the movie entry using the title and the startYear
    movieEntry = df.query('primaryTitle == "%s" and startYear == %d' % (ti, year)).\
                    iloc[0].to_dict()

    movieEntry["directors_names"] = movieEntry["directors_names"].split('/')
    movieEntry["writers_names"] = movieEntry["writers_names"].split('/')
    movieEntry["cast_name"] = movieEntry["cast_name"].split('/')
    movieEntry["genres"] = movieEntry["genres"].split(',')

    # entry_feat = convert_to_feat(movieEntry)
    seen_movies.append(longTitle)

    return movieEntry

def feedback(request):
    global df, titles, current_user_feat_X, current_user_feat_Y
    user_movie_entry = defaultdict(list)
    likeChoice = request.POST.get('likeChoice', False)
    # print(request.POST.get("genres", "N/A")[1])
    feat_list = ["directors_names", "writers_names", "cast_name", "genres"]
    for feat in feat_list:
        temp = request.POST.getlist(feat, "N/A")
        if temp != "N/A":
            user_movie_entry[feat] = temp
        print(" ".join(temp) if temp != "N/A" else "N/A")
    user_movie_entry["startYear"] = [request.POST.get("year")]
    print(request.POST.get("year"))
    user_movie_entry["rating"] = [request.POST.get("rating")]
    print(request.POST.get("rating"))

    user_feat = convert_to_feat(user_movie_entry)
    current_user_feat_X.append(user_feat)
    current_user_feat_Y.append(likeChoice == "like")
    if len(current_user_feat_X) > 4:
        train(np.array(current_user_feat_X),np.array(current_user_feat_Y))

    movieEntry = fetchFeatures()
    return render(request, "home.html", {'titles': titles, "movieData": movieEntry})


"""
helper functions to do feature matching/cleaning (copied from previous notebook)
"""
# column indices
col = {'tconst':0,
    'primaryTitle':1,
    'originalTitle':2,
    'titlesfromUS/UK':3,
    'startYear':4,
    'region':5,
    'genres':6,
    'directors_name':7,
    'writers_name':8,
    'cast_name':9,
    'averageRating':10,
    'numVotes':11}

def get_column(matrix, i):
    return [row[i] for row in matrix]

def features_construction():
    if not os.path.exists("feat_to_index.pkl") or not os.path.exists("index_to_feat.pkl"):
        # read data
        df1 = pd.read_csv('IMDB_Final_Movies.csv')
        # data in numpy
        data = df1.values
        # create list for one hot encoding
        cast_list = list(set(["c_" + j for i in [x.split('/') for x in get_column(data,col["cast_name"])] for j in i]))
        director_list = list(set(["d_" + j for i in [str(x).split('/') for x in get_column(data,col["directors_name"])] for j in i]))
        writers_list = list(set(["w_" + j for i in [str(x).split('/') for x in get_column(data,col["writers_name"])] for j in i]))
        genre_list = list(set(["g_" + j for i in [str(x).split(',') for x in get_column(data,col["genres"])] for j in i]))
        year_list = list(np.arange(1890,2020,10))

        # create list for one hot encoding
        onehot_list = list(set(cast_list + director_list + writers_list + genre_list + year_list))
        onehot_indices = list(np.arange(0,len(onehot_list),1))
        onehot_feat_to_index = dict(zip(onehot_list,onehot_indices))
        onehot_index_to_feat = dict(zip(onehot_indices,onehot_list))

        f = open("feat_to_index.pkl","wb")
        pickle.dump(onehot_feat_to_index,f)
        f.close()

        f = open("index_to_feat.pkl","wb")
        pickle.dump(onehot_index_to_feat,f)
        f.close()
    else:
        onehot_feat_to_index = pickle.load(open("feat_to_index.pkl","rb"))
        onehot_index_to_feat = pickle.load(open("index_to_feat.pkl","rb"))
    return onehot_feat_to_index, onehot_index_to_feat

def convert_to_feat(movie_entry):
    global onehot_feat_to_index
    feat = np.zeros(len(onehot_feat_to_index))
    # cast
    for c in movie_entry["cast_name"]:
        feat[onehot_feat_to_index[c]] = 1
    # director
    for d in movie_entry["directors_names"]:
        feat[onehot_feat_to_index[d]] = 1
    # writer
    for w in movie_entry["writers_names"]:
        feat[onehot_feat_to_index[w]] = 1
    # genre
    for g in movie_entry["genres"]:
        feat[onehot_feat_to_index[g]] = 1
    # year
    if movie_entry["startYear"][0] != "no":
        decade = int(int(movie_entry["startYear"][0])/10) * 10
        feat[onehot_feat_to_index[decade]] = 1

    return feat

def train(X,Y):
    global onehot_index_to_feat
    from sklearn.tree import DecisionTreeClassifier
    from scipy.sparse import lil_matrix
    print(X.shape[1])
    X = lil_matrix(X)
    clf = DecisionTreeClassifier()
    clf.fit(X,Y)
    print(len(clf.feature_importances_))
    for i,d in enumerate(clf.feature_importances_):
        if d != 0:
            print(onehot_index_to_feat[i], d)
