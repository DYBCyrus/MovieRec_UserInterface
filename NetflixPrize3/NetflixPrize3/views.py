from django.shortcuts import render
from django.http import JsonResponse
from django.core import serializers
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import os
import json

df = pd.DataFrame()
titles = []
sorted_titles = []

def button(request):
    # print(os.getcwd())
    global df
    global titles, sorted_titles
    if len(titles) > 0:
        movieEntry = fetchFeatures()
        return render(request, 'home.html', {'titles':titles, 'movieData': movieEntry})
    df = pd.read_csv('IMDB_Final_Movies.csv', delimiter=',')
    title = df['primaryTitle'].tolist()
    year = df['startYear'].tolist()
    sorted_titles = json.load(open("sorted_movies_for_genres.json"))
    for (i,j) in zip(title,year):
        titles.append(i + '(' + str(int(j)) + ')')
    movieEntry = fetchFeatures()
    return render(request, 'home.html', {'titles': titles, 'movieData': movieEntry})

def fetchFeatures():
    global df
    global titles, sorted_titles
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

    return movieEntry

def feedback(request):
    global titles
    global df
    likeChoice = request.POST.get('likeChoice', False)
    lol = ""
    if likeChoice == "like":
        lol = "so you like it"
    else:
        lol = "so you don't like it"
    # print(request.POST.get("genres", "N/A")[1])
    feat_list = ["directors_names", "writers_names", "cast_name", "genres"]
    for feat in feat_list:
        temp = request.POST.getlist(feat, "N/A")
        print(" ".join(temp) if temp != "N/A" else "N/A")
    print(request.POST.get("year"))
    print(request.POST.get("rating"))

    movieEntry = fetchFeatures()
    return render(request, "home.html", {'titles': titles, "lol": lol, "movieData": movieEntry})
