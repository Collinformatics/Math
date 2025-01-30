def KLDivergence(self, P, Q, printProb, fixedSeq, scaler):
    print('===================================== KL Divergence '
          '=====================================')
    P.columns = Q.columns
    if printProb:
        print(f'Baseline Probability Distribution:\n{Q}\n\n\n'
              f'Probability Distribution:\n{P}\n\n')

    divergence = pd.DataFrame(0, columns=Q.columns, index=[fixedSeq], dtype=float)
    divergenceMatrix = pd.DataFrame(0, columns=Q.columns, index=Q.index, dtype=float)

    for position in Q.columns:
        p = P.loc[:, position]
        q = Q.loc[:, position]
        divergence.loc[fixedSeq, position] = (
            np.sum(np.where(p != 0, p * np.log2(p / q), 0)))

        for residue in Q.index:
            initial = Q.loc[residue, position]
            final = P.loc[residue, position]
            if initial == 0 or final == 0:
                divergenceMatrix.loc[residue, position] = 0
            else:
                divergenceMatrix.loc[residue, position] = final * np.log2(final / initial)

    # Scale the values
    if scaler is not None:
        for position in Q.columns:
            divergenceMatrix.loc[:, position] = (divergenceMatrix.loc[:, position] *
                                                 scaler.loc[position, 'ΔEntropy'])

    print(f'{silver}KL Divergence:{pink} Fixed Final Sort - {fixedSeq}{resetColor}\n'
          f'{divergence}\n\n\n{silver}Divergency Matrix:{pink} Fixed Final Sort - {fixedSeq}'
          f'{resetColor}\n{divergenceMatrix.round(4)}\n\n')

    return divergenceMatrix, divergence
