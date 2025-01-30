# Calculate the Kullback–Leibler (KL) divergence

import numpy as np
import pandas as pd



# ===================================== Create Data ======================================
probDist1 = pd.DataFrame(np.random.rand(5, 5),
                         index=['A', 'B', 'C', 'D', 'E'],
                         columns=['V', 'W', 'X', 'Y', 'Z'])
probDist1 = probDist1.div(probDist1.sum(axis=0), axis=1) # Normalize values
probDist2 = pd.DataFrame(np.random.rand(5, 5),
                         index=['A', 'B', 'C', 'D', 'E'],
                         columns=['V', 'W', 'X', 'Y', 'Z'])
probDist2 = probDist2.div(probDist2.sum(axis=0), axis=1)
probScaler = pd.DataFrame([1.21, 0.87, 2.37, 0.98, 0.54])
probLabel = 'Random Dataset'



# =================================== Define Function ====================================
def KLDivergence(P, Q, datasetTag, scaler):
    print('================================= KL Divergence '
          '=================================')
    P.columns = Q.columns
    print(f'True Probability Distribution:\n'
          f'{P}\n\n'
          f'Model Probability Distribution:\n'
          f'{Q}\n\n')

    # modelDistize: Dataframes
    divergence = pd.DataFrame(0, 
                              columns=Q.columns, 
                              index=[datasetTag], 
                              dtype=float)
    divergenceMatrix = pd.DataFrame(0, 
                                    columns=Q.columns, 
                                    index=Q.index, 
                                    dtype=float)

    # Calculate: Divergence
    for column in Q.columns:
        p = P.loc[:, column]
        q = Q.loc[:, column]
        divergence.loc[datasetTag, column] = (
            np.sum(np.where(p != 0, p * np.log2(p / q), 0)))

        for row in Q.index:
            trueDist = P.loc[row, column]
            modelDist = Q.loc[row, column]

            # Do not allow NaN
            if modelDist == 0 or trueDist == 0:
                divergenceMatrix.loc[row, column] = 0 # Set NaN to 0
            else:
                divergenceMatrix.loc[row, column] = (trueDist * 
                                                     np.log2(trueDist / modelDist))

    # Scale the values
    if scaler is not None:
        for index, column in enumerate(Q.columns):
            divergenceMatrix.loc[:, column] = (divergenceMatrix.loc[:, column] *
                                               scaler.iloc[index, 0])

    print(f'KL Divergence: {datasetTag}\n'
          f'{divergence}\n\n\n'
          f'Divergency Matrix: {datasetTag}'
          f'\n{divergenceMatrix.round(4)}\n\n')

    return divergenceMatrix, divergence



# ==================================== Evaluate Data =====================================
KLDivergence(P=probDist1,
             Q=probDist2,
             datasetTag=probLabel,
             scaler=probScaler)
