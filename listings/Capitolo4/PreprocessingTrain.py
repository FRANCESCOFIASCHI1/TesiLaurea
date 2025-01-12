X_train_final = []
dfSegment = pd.read_csv("data/segments.csv", index_col="timestamp")

for channel in dfSegment["channel"].unique():
    # Itera su ogni segmento unico per il canale corrente
    for segment in dfSegment[dfSegment["channel"] == channel]["segment"].unique():
        mask = (dfSegment["train"] == 1) & (dfSegment["channel"] == channel) & (dfSegment["segment"] == segment)
        
        # Filtra i dati in base alla maschera
        X_trainS = dfSegment.loc[mask, "value"] 
        
        # Suddividi in sottoliste di STEP elementi
        for i in range(0, len(X_trainS) - STEP + 1, STEP):
            sublist = X_trainS[i:i + STEP]
            X_train_final.append(sublist)

# Converti la lista in un numpy array
X_train = np.array(X_train_final)

# Reshape per ottenere la shape desiderata
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_train = X_train.transpose(0, 2, 1)