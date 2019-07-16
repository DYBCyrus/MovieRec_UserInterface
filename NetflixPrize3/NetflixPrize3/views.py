from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import os
import json



def button(request):
    # print(os.getcwd())
    global df
    df = pd.read_csv('IMDB_Final_Movies.csv', delimiter=',')
    title = df['primaryTitle'].tolist()
    year = df['startYear'].tolist()
    titles = []
    for (i,j) in zip(title,year):
        if not np.isnan(j):
            titles.append(i + '(' + str(int(j)) + ')')
        else:
            titles.append(i + '(N/A)')
    print(titles[0])
    return render(request, 'home.html', {'titles': titles})

def fetchFeatures(request):
    longTitle = request.POST.get('movie_title', False)
    title = ""
    if longTitle:
        title = longTitle.split('(')
        year = int(title[1].split(')')[0])
        title = title[0]
    # Search the movie entry using the title and the startYear
    movieEntry = df.query('primaryTitle == title and startYear == year')
    directors = movieEntry.iloc[0]["directors_names"]
    # return JsonResponse({"directors":directors})
    return render(request, "home.html", {"directors",directors})
