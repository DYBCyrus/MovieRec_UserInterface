import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib import rcParams

def plot_features(sortedLogFeat, picName, pltTitle):
    xlabels = []
    coefs = []
    colors = []

    for i in range( len(sortedLogFeat) if len(sortedLogFeat) < 15 else 15 ):
        xlabels.append(sortedLogFeat[i][0])
        coefs.append(abs(sortedLogFeat[i][1]))
        if sortedLogFeat[i][1] >= 0:
            colors.append('C0')
        else:
            colors.append("#ffa600")

    y_pos = np.arange(len(xlabels))
    # plt.figure(figsize=(10,12))
    plt.bar(y_pos, coefs, align='center', alpha=0.5, color=colors)
    rcParams.update({'figure.autolayout': True})
    plt.xticks(y_pos, xlabels, rotation=45, horizontalalignment='right')
    plt.ylabel('Coefficient')
    plt.xlabel("Features")
    plt.title('Main Contributing Features of ' + pltTitle)
    # plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig('static/NetflixPrize3/Contr' + picName + '.png')
    plt.close()

    # xlabels = []
    # coefs = []
    # # Plot negative
    # start = len(sortedLogFeat) - 1
    # end = start - 10
    # for i in range(start, end, -1):
    #     xlabels.append(sortedLogFeat[i][0])
    #     coefs.append(sortedLogFeat[i][1])
    # y_pos = np.arange(len(xlabels))
    # # plt.figure(figsize=(10,12))
    # plt.bar(y_pos, coefs, align='center', alpha=0.5, color="#ffa600")
    # rcParams.update({'figure.autolayout': True})
    # plt.xticks(y_pos, xlabels, rotation=45, horizontalalignment='right')
    # plt.ylabel('Coefficient')
    # plt.xlabel("Features")
    # plt.title('Main Negative Features of ' + pltTitle)
    # # plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    # plt.tight_layout()
    # plt.savefig('static/NetflixPrize3/Neg' + picName + '.png')
    # plt.close()
