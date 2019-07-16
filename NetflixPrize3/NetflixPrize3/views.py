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
    print(titles[0])
    return render(request, 'home.html', {'titles': titles})

def fetchFeatures(request):
    global df
    global titles
    longTitle = request.POST.get('title', False)
    title = ""
    year = 0
    if longTitle:
        title = longTitle.split('(')
        year = int(title[1].split(')')[0])
        title = title[0]
    # Search the movie entry using the title and the startYear
    movieEntry = df.query('primaryTitle == "%s" and startYear == %d' % (title, year)).\
                    iloc[0].to_dict()
    directors = movieEntry["directors_names"].split('/')
    writers = movieEntry["writers_names"].split('/')
    cast = movieEntry["cast_name"].split('/')
    # return JsonResponse({"directors":directors})

    return render(request, "home.html", {'titles': titles, "directors":directors})
