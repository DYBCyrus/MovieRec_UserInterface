from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import os
import json

def button(request):
    # print(os.getcwd())
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

