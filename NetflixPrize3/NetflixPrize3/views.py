from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
import random

def button(request):
    return render(request, 'home.html')

