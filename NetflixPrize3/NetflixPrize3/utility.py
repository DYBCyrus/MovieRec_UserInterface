import matplotlib.pyplot as plt
import numpy as np

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
    plt.figure(figsize=(10,12))
    plt.bar(y_pos, coefs, align='center', alpha=0.5)
    plt.xticks(y_pos, xlabels, rotation=45)
    plt.ylabel('Feature Coefficient')
    plt.title('Main Contributing Features')

    plt.savefig('static/NetflixPrize3/MainFeatures.png')
    plt.close()