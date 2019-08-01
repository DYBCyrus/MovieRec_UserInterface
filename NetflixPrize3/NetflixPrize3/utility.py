import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

def plot_features(sortedLogFeat):
    i = 0
    xlabels = []
    coefs = []
    # sortedLogFeat[i][1] > 0 and
    while i < 5:
        xlabels.append(sortedLogFeat[i][0])
        coefs.append(sortedLogFeat[i][1])
        i += 1
    y_pos = np.arange(len(xlabels))
    # plt.figure(figsize=(10,12))
    plt.bar(y_pos, coefs, align='center', alpha=0.5)
    rcParams.update({'figure.autolayout': True})
    plt.xticks(y_pos, xlabels, rotation=45, horizontalalignment='right')
    plt.ylabel('Coefficient')
    plt.xlabel("Features")
    plt.title('Main Contributing Features')
    # plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.savefig('static/NetflixPrize3/MainFeatures.png')
    plt.close()
