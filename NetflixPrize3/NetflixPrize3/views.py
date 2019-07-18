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

def button(request):
    # print(os.getcwd())
    global df
    df = pd.read_csv('IMDB_Final_Movies.csv', delimiter=',')
    title = df['primaryTitle'].tolist()
    year = df['startYear'].tolist()
    global titles
    for (i,j) in zip(title,year):
        if not np.isnan(j):
            titles.append(i + '(' + str(int(j)) + ')')
        else:
            titles.append(i + '(N/A)')
    return render(request, 'home.html', {'titles': titles})

def fetchFeatures(request):
    global df
    global titles
    longTitle = request.POST.get('title', False)
    title = ""
    year = 0
    if longTitle:
        title = longTitle.split('(')
        ti = title[0]
        if (df.query('primaryTitle == "%s"' % (ti))).empty:
            return render(request, "home.html", {'titles': titles, "titleInvalid":True})
        if '(N/A)' not in longTitle:
            year = int(title[1].split(')')[0])
    else:
        return render(request, "home.html", {'titles': titles, "titleInvalid":True})
    if year != 0:
        # Search the movie entry using the title and the startYear
        movieEntry = df.query('primaryTitle == "%s" and startYear == %d' % (ti, year)).\
                        iloc[0].to_dict()
    else:
        movieEntry = df.query('primaryTitle == "%s"' % (ti)).iloc[0].to_dict()

    movieEntry["directors_names"] = movieEntry["directors_names"].split('/')
    movieEntry["writers_names"] = movieEntry["writers_names"].split('/')
    movieEntry["cast_name"] = movieEntry["cast_name"].split('/')

    # return JsonResponse({"directors":directors})

    return render(request, "home.html", {'titles': titles, "movieData":movieEntry})

def feedback(request):
    global titles
    global df
    likeChoice = request.POST.get('likeChoice', False)
    lol = ""
    if likeChoice == "like":
        lol = "so you like it"
    else:
        lol = "so you don't like it"
    return render(request, "home.html", {'titles': titles, "lol":lol})
