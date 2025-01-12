X_test_final = []
y_test_final = []

for channel in dfSegment["channel"].unique():
    for segment in test_data[test_data["channel"] == channel]["segment"].unique():

        mask = (test_data["channel"] == channel) & (test_data["segment"] == segment)
        X_testS = test_data.loc[mask, "value"]
        y_testS = test_data.loc[mask, "anomaly"]
        
        for i in range(0, len(X_testS) - STEP + 1, STEP):
            X_test_final.append(X_testS[i:i + STEP])
            y_test_final.append(y_testS[i])

X_test = np.array(X_test_final)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_test = X_test.transpose(0, 2, 1)
y_test = np.array(y_test_final)