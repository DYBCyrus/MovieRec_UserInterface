from django.shortcuts import render
from django.http import JsonResponse
from django.core import serializers
import numpy as np
import random
import pandas as pd
import math
import os
import operator
import json
import pickle
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import lil_matrix
from sklearn.tree.export import export_text
from sklearn import preprocessing
from . import utility

df = pd.DataFrame()
df1 = pd.DataFrame()
titles = []
sorted_titles = []
onehot_feat_to_index = {}
onehot_index_to_feat = {}
seen_movies = ["dummy"]
# movies that user has chosen
current_user_feat_X = []
current_user_feat_Y = []
# all movies feature vector
movies_feat = lil_matrix(0)
# index to movie title + year
index_to_movie_title_year = {}
# index to movie imdb-rating,imdb-numvote,critic-rating,critic-numvote
index_to_movie_rating_numVotes = {}
dislikeExists = False
likeExists = False
twoSelectionsExist = False
logistic = False
min_rating = max_rating = min_numVotes = max_numVotes = mean_rating = mean_numVotes = 0

def button(request):
    # print(os.getcwd())
    global df, titles, sorted_titles, onehot_feat_to_index, onehot_index_to_feat, df1
    global index_to_movie_title_year, index_to_movie_rating_numVotes
    onehot_feat_to_index, onehot_index_to_feat, index_to_movie_title_year, \
        index_to_movie_rating_numVotes = features_construction()
    if len(titles) > 0:
        movieEntry = fetchFeatures()
        return render(request, 'home.html', {'titles':titles, 'movieData': movieEntry})
    # df = pd.read_csv('IMDB_Meta_Combined_Final.csv')
    title = df['primaryTitle'].tolist()
    year = df['startYear'].tolist()
    sorted_titles = json.load(open("sorted_movies_for_genres.json"))
    for (i,j) in zip(title,year):
        titles.append(i + '(' + str(int(j)) + ')')
    movieEntry = fetchFeatures()
    dir_path = os.getcwd()
    os.makedirs(os.path.join(dir_path,'log'), exist_ok=True)
    # Create log file here
    # Obtain the time and create a string for file name

    return render(request, 'home.html', {'titles': titles, 'movieData': movieEntry})

def fetchFeatures(longTitle="dummy"):
    global df, titles, sorted_titles, seen_movies, df1
    while longTitle in seen_movies:
        # longTitle = request.POST.get('title', False)
        gen = random.sample(sorted_titles.keys(),1)[0]
        rand_sample = random.sample(sorted_titles[gen][:100],1)
        longTitle = rand_sample[0][0] + "(" + rand_sample[0][1] + ")"
    title = ""

    # Append title to the file
    print()
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
    movieEntry_rate_vote = df1.query('primaryTitle == "%s" and startYear == %d' % (ti, year)).\
                    iloc[0].to_dict()

    movieEntry["directors_names"] = movieEntry["directors_names"].split('/')
    movieEntry["writers_names"] = movieEntry["writers_names"].split('/')
    movieEntry["cast_name"] = movieEntry["cast_name"].split('/')
    movieEntry["genres"] = movieEntry["genres"].split(',')

    movieEntry["averageRating"] = movieEntry_rate_vote["averageRating"]
    movieEntry["numVotes"] = movieEntry_rate_vote["numVotes"]
    movieEntry["metascore"] = movieEntry_rate_vote["metascore"]
    movieEntry["critics_reviews_count"] = movieEntry_rate_vote["critics_reviews_count"]

    seen_movies.append(longTitle)

    return movieEntry

def feedback(request):
    global df, titles, current_user_feat_X, current_user_feat_Y, \
            dislikeExists, likeExists, twoSelectionsExist, logistic, df1
    user_movie_entry = defaultdict(list)
    likeChoice = request.POST.get('likeChoice', False)
    # print(request.POST.get("genres", "N/A")[1])
    feat_list = ["directors_names", "writers_names", "cast_name", "genres"]
    for feat in feat_list:
        temp = request.POST.getlist(feat, "N/A")
        if temp != "N/A":
            user_movie_entry[feat] = temp
        print(" ".join(temp) if temp != "N/A" else "N/A")
    user_movie_entry["startYear"] = [request.POST.get("year","no")]
    print(request.POST.get("year"))
    user_movie_entry["rating"] = [request.POST.get("rating","no")]
    print(request.POST.get("rating"))
    user_movie_entry["numVotes"] = [request.POST.get("numVotes", "no")]
    print(request.POST.get("numVotes"))

    user_feat = convert_to_feat(user_movie_entry, int(likeChoice == "like"))
    current_user_feat_X.append(user_feat)
    current_user_feat_Y.append(likeChoice == "like")
    if not twoSelectionsExist:
        dislikeExists = likeChoice == "dislike" or dislikeExists
        likeExists = likeChoice == "like" or likeExists
        twoSelectionsExist = likeExists and dislikeExists

    ex = None
    recommended = None
    if len(current_user_feat_X) > 4 and twoSelectionsExist:
        ex, recommended = train(np.array(current_user_feat_X),np.array(current_user_feat_Y))

    # lime = False
    if request.POST.get('fetch', 'Random') == 'Recommend' and recommended:
        movieEntry = fetchFeatures(recommended)
        logistic = True
        # lime = True
    else:
        movieEntry = fetchFeatures()
    return render(request, "home.html", {'titles': titles, "movieData": movieEntry,\
        "explanation": ex, "logistic":logistic})


"""
helper functions to do feature matching/cleaning (copied from previous notebook)
"""
# column indices
col = {'tconst':0,
    'primaryTitle':1,
    'startYear':2,
    'genres':3,
    'directors_names':4,
    'writers_names':5,
    'cast_name':6,
    'averageRating':7,
    'numVotes':8,
    'metascore':9,
    'critics_reviews_count':10,
    'description':11}

def get_column(matrix, i):
    return [row[i] for row in matrix]

def features_construction():
    global movies_feat, col, mean_rating, mean_numVotes, df, df1
    # read data
    df = pd.read_csv('Combined_Dataset_Final.csv')
    df1 = df.copy()
    df['numVotes'] = df['numVotes'].apply(lambda x: math.log(x,10)).copy()

    '''
    reference: https://chrisalbon.com/python/data_wrangling/pandas_normalize_column/
    '''
    x = df[['numVotes']].values.astype(float)
    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()
    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(x)
    # Run the normalizer on the dataframe
    df_normalized = pd.DataFrame(x_scaled, columns=['norm_numVotes'])

    df['numVotes'] = df_normalized['norm_numVotes'].values
    
    # data in numpy
    data = df.values
    if not os.path.exists("feat_to_index.pkl") or not os.path.exists("index_to_feat.pkl"):
        # create list for one hot encoding
        cast_list = list(set(["c_" + j for i in [x.split('/') for x in get_column(data,col["cast_name"])] for j in i]))
        director_list = list(set(["d_" + j for i in [str(x).split('/') for x in get_column(data,col["directors_names"])] for j in i]))
        writers_list = list(set(["w_" + j for i in [str(x).split('/') for x in get_column(data,col["writers_names"])] for j in i]))
        genre_list = list(set(["g_" + j for i in [str(x).split(',') for x in get_column(data,col["genres"])] for j in i]))
        year_list = list(np.arange(1890,2020,10))

        # create list for one hot encoding
        onehot_list = list(set(cast_list + director_list + writers_list + genre_list + year_list)) \
            + ["rating"] + ["numVotes"]
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

    index_to_movie = {}
    index_to_rate_vote = {}

    mean_rating = df['averageRating'].mean()
    mean_numVotes = df['numVotes'].mean()

    if movies_feat.shape == (1,1):
        movies_feat = lil_matrix((len(data),len(onehot_feat_to_index)))
        for i in range(len(data)):
            m = data[i]
            index_to_movie[i] = m[col['primaryTitle']] + '(' + str(int(m[col['startYear']])) + ')'
            index_to_rate_vote[i] = [df1.iloc[i,col['averageRating']], df1.iloc[i,col['numVotes']],\
                df1.iloc[i,col['metascore']], df1.iloc[i,col['critics_reviews_count']]]
            # cast
            for c in str(m[col["cast_name"]]).split("/"):
                movies_feat[i,onehot_feat_to_index["c_"+c]] = 1
            # director
            for d in str(m[col["directors_names"]]).split("/"):
                movies_feat[i,onehot_feat_to_index["d_"+d]] = 1
            # writer
            for w in str(m[col["writers_names"]]).split("/"):
                movies_feat[i,onehot_feat_to_index["w_"+w]] = 1
            # genre
            for g in str(m[col["genres"]]).split(","):
                movies_feat[i,onehot_feat_to_index["g_"+g]] = 1
            # year
            decade = int(m[col["startYear"]]/10) * 10
            movies_feat[i,onehot_feat_to_index[decade]] = 1

            # rating
            movies_feat[i,len(onehot_feat_to_index)-2] = m[col["averageRating"]]
            # numVotes
            movies_feat[i,len(onehot_feat_to_index)-1] = m[col["numVotes"]]

    if not os.path.exists("index_to_movie.pkl") or not os.path.exists("index_to_rate_vote.pkl"):
        f = open("index_to_movie.pkl","wb")
        pickle.dump(index_to_movie,f)
        f.close()
        f = open("index_to_rate_vote.pkl","wb")
        pickle.dump(index_to_rate_vote,f)
        f.close()
    else:
        index_to_movie = pickle.load(open("index_to_movie.pkl","rb"))
        index_to_rate_vote = pickle.load(open("index_to_rate_vote.pkl","rb"))

    return onehot_feat_to_index, onehot_index_to_feat, index_to_movie, index_to_rate_vote

def convert_to_feat(movie_entry, label):
    global onehot_feat_to_index, current_user_feat_X, current_user_feat_Y, mean_rating, mean_numVotes
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
    # rating
    magicRating = mean_rating
    if movie_entry["rating"][0] != "no":
        feat[len(onehot_feat_to_index)-2] = float(movie_entry["rating"][0])
    else:
        satisfied_Y = np.where(np.array(current_user_feat_Y) == label)[0]
        feat[len(onehot_feat_to_index)-2] = \
            np.mean(np.array(current_user_feat_X)[satisfied_Y][:,len(onehot_feat_to_index)-2]) \
                if len(satisfied_Y) > 0 else magicRating
    # numVotes
    magicNumVotes = mean_numVotes
    if movie_entry["numVotes"][0] != "no":
        feat[len(onehot_feat_to_index)-1] = math.log(float(movie_entry["numVotes"][0]),10)
    else:
        satisfied_Y = np.where(np.array(current_user_feat_Y) == label)[0]
        feat[len(onehot_feat_to_index)-1] = \
            np.mean(np.array(current_user_feat_X)[satisfied_Y][:,len(onehot_feat_to_index)-1]) \
                if len(satisfied_Y) > 0 else magicNumVotes

    return feat

def train(X,Y):
    global onehot_index_to_feat, movies_feat, index_to_movie_title_year, index_to_movie_rating_numVotes
    global min_rating, max_rating, min_numVotes, max_numVotes, seen_movies
    X = lil_matrix(X)

    print("Start training LogisticRegression")
    logClf = LogisticRegression(random_state = 0, max_iter=100, solver='liblinear', penalty='l2').fit(X, Y)

    logPreds = logClf.predict_proba(movies_feat)
    log_ascending_recommended_movie = np.argsort(logPreds[:,-1])
    for log_recommended_movie in log_ascending_recommended_movie[::-1]:
        if index_to_movie_title_year[log_recommended_movie] not in seen_movies:
            break

    logCoef = logClf.sparsify().coef_
    """
    fetchFeatures passes a dictionary
    """
    featToCoef = {}
    target_movie = movies_feat[log_recommended_movie,:].toarray()[0].tolist()
    for k,v in enumerate(target_movie):
        if v != 0:
            if type(onehot_index_to_feat[k]) == str and onehot_index_to_feat[k].startswith('c_'):
                c = onehot_index_to_feat[k].split('c_')[1]
                featToCoef["Cast_" + c] = logCoef[0, k]
            elif type(onehot_index_to_feat[k]) == str and onehot_index_to_feat[k].startswith('d_'):
                d = onehot_index_to_feat[k].split('d_')[1]
                featToCoef["Director_" + d] = logCoef[0, k]
            elif type(onehot_index_to_feat[k]) == str and onehot_index_to_feat[k].startswith('w_'):
                w = onehot_index_to_feat[k].split('w_')[1]
                featToCoef["Writer_" + w] = logCoef[0, k]
            elif type(onehot_index_to_feat[k]) == str and onehot_index_to_feat[k].startswith('g_'):
                g = onehot_index_to_feat[k].split('g_')[1]
                featToCoef["Genre_" + g] = logCoef[0, k]
            elif k == len(onehot_feat_to_index)-1:
                featToCoef["NumVotes_{}".format(index_to_movie_rating_numVotes[log_recommended_movie][1])] \
                    = logCoef[0, k] * v
            elif k == len(onehot_feat_to_index)-2:
                featToCoef["Rating_{}".format(index_to_movie_rating_numVotes[log_recommended_movie][0])] \
                    = logCoef[0, k] * v
            else:
                featToCoef["Decade_{}".format(onehot_index_to_feat[k])] = logCoef[0, k]

    sortedLogFeat = sorted(featToCoef.items(), key=operator.itemgetter(1), reverse=True)

    utility.plot_features(sortedLogFeat)

    print(index_to_movie_title_year[log_recommended_movie])

    return "explanation", index_to_movie_title_year[log_recommended_movie]
