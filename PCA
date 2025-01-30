from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import string



# ===================================== User Inputs ======================================
inNumLabels = 1000
inNumVariables = 40
inNumberPCs = 2
inFigureTitle = 'PCA Analysis'
inFigureSize = (9, 8)



# =================================== Define Functions  ===================================
def PCA(data, numberOfPCs, title, figSize):
    print('====================================== PCA '
          '======================================')
    from sklearn.decomposition import PCA

    # Initialize lists to  of clustered labels
    selectedLabels = []
    selectedValues = []
    rectangles = []


    # Define component labels
    pcaHeaders = []
    for componetNumber in range(1, numberOfPCs + 1):
        pcaHeaders.append(f'PC{componetNumber}')
    headerCombinations = list(combinations(pcaHeaders, 2))


    # # Cluster the datapoints
    # Step 1: Apply PCA on the standardized data
    pca = PCA(n_components=numberOfPCs)  # Adjust the number of components as needed
    dataPCA = pca.fit_transform(data)
    # loadings = pca.components_.T

    # Step 2: Create a DataFrame for PCA results
    dataPCA = pd.DataFrame(dataPCA, columns=pcaHeaders, index=data.index)
    pd.set_option('display.max_rows', 10)
    print(f'PCA Data: # of componets = {numberOfPCs}\n'
          f'{dataPCA}\n\n')

    # Step 3: Print explained variance ratio
    # Evaluate variance captured by each component
    varRatio = pca.explained_variance_ratio_ * 100
    varRatio1 = np.round(varRatio[0], 3)
    varRatio2 = np.round(varRatio[1], 3)
    print(f'Explained Variance Ratio: {varRatio1}%, {varRatio2}%\n\n')


    # Plot the data
    for componets in headerCombinations:
        fig, ax = plt.subplots(figsize=figSize)
        plt.scatter(dataPCA[componets[0]], dataPCA[componets[1]],
                    c='#CC5500', edgecolor='black')
        plt.xlabel(f'Principal Component {componets[0][-1]}',
                   fontsize=16)
        plt.ylabel(f'Principal Component {componets[1][-1]}',
                   fontsize=16)
        plt.title(f'{title}\nVariance Ratio: {varRatio1}%, {varRatio2}%',
                  fontsize=18, fontweight='bold')


        # Set tick parameters
        ax.tick_params(axis='both', which='major', length=4,
                       labelsize=13, width=1.5)

        # Set the thickness of the figure border
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.5)


        fig.canvas.mpl_connect('key_press_event', pressKey)
        fig.tight_layout()
        plt.show()



def makeList(numberOfElements):
    labels = []
    for index in range(numberOfElements):
        label = ''
        addLabel = False
        while not addLabel:
            for _ in range(3):
                label += string.ascii_uppercase[random.randint(0, 25)]
            if label not in labels:
                labels.append(label)
                addLabel = True
    return labels



def pressKey(event):
    if event.key == 'escape':
        plt.close()



# ===================================== Create Data ======================================
values = pd.DataFrame(np.random.rand(inNumLabels, inNumVariables),
                      index=makeList(inNumLabels),
                      columns=list(makeList(inNumVariables)))



# ==================================== Evaluate Data =====================================
PCA(data=values, numberOfPCs=inNumberPCs, title=inFigureTitle, figSize=inFigureSize)
