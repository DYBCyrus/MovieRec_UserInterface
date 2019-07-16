from django.shortcuts import render
from django.http import JsonResponse
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
    if year != 0:
        # Search the movie entry using the title and the startYear
        movieEntry = df.query('primaryTitle == "%s" and startYear == %d' % (ti, year))
    else:
        movieEntry = df.query('primaryTitle == "%s"' % (ti))
    directors = movieEntry.iloc[0]["directors_names"]
    # return JsonResponse({"directors":directors})
    return render(request, "home.html", {'titles': titles, "directors":directors})
