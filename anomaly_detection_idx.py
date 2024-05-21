def anomaly_detection_idx(Data, Group_Anomaly, idx):
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.covariance import MinCovDet
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import matplotlib.pyplot as plt
    from scipy.signal import filtfilt
    import rrcf
    #from scipy.io import loadmat

    # Define the indices of the training dataset
    train_start = 1700 
    train_end = 2000    

    # Define the window length
    windows_length = min(150, len(Data) - 1) 

    # Extract the training data
    data_train = Data.iloc[train_start:train_end, Group_Anomaly['Alma_OUT'].iloc[idx]] 

    # Standardize the data
    scaler = StandardScaler() # Initialize the scaler
    data_train = scaler.fit_transform(data_train) # Standardize the data

    data_test = Data.iloc[:, Group_Anomaly['Alma_OUT'].iloc[idx]]  # Extract the test data
    data_test = scaler.fit_transform(data_test) # Standardize the data

    # RRCF
    forest = rrcf.RCTree()
    for index, point in enumerate(data_train): 
        forest.insert_point(point, index)

    # Insert test data into the tree and calculate CoDisp
    score_rrcf = []
    for index, point in enumerate(data_test, start=len(data_train)): 
        forest.insert_point(point, index)
        score_rrcf.append(forest.codisp(index))
    #score_rrcf = MinMaxScaler().fit_transform(np.array(score_rrcf).reshape(-1, 1)).flatten() # Normalize the scores
    smoothed_rrcf = filtfilt(np.ones(windows_length), windows_length, score_rrcf)
    movingavg_rrcf = pd.Series(score_rrcf).rolling(windows_length).mean().to_numpy()

    # IF
    iforest = IsolationForest(n_estimators=250, max_samples=200)
    iforest.fit(data_train)
    score_if = -iforest.decision_function(data_test)
    smoothed_if = filtfilt(np.ones(windows_length), windows_length, score_if)
    movingavg_if = pd.Series(score_if).rolling(windows_length).mean().to_numpy()

    # LOF
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof.fit(data_train)
    score_lof = -lof.decision_function(data_test) # Invert the scores to match the score of MATLAB Algorithm
    smoothed_lof = filtfilt(np.ones(windows_length), windows_length, score_lof)
    movingavg_lof = pd.Series(score_lof).rolling(windows_length).mean().to_numpy()

    # OCSVM
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale')
    ocsvm.fit(data_train)
    score_ocsvm = -ocsvm.score_samples(data_test)
    smoothed_ocsvm = filtfilt(np.ones(windows_length), windows_length, score_ocsvm)
    movingavg_ocsvm = pd.Series(score_ocsvm).rolling(windows_length).mean().to_numpy()

    # MAHAL
    mcd = MinCovDet()
    mcd.fit(data_train)
    mu = mcd.location_
    sigma = mcd.covariance_
    if np.linalg.det(sigma) != 0:
        sTest_mahal = distance.cdist(data_test, mu.reshape(1, -1), 'mahalanobis', VI=np.linalg.inv(sigma))
        sTest_mahal = sTest_mahal.flatten()  # Add this line to flatten the array
        smoothed_mahal = filtfilt(np.ones(windows_length), windows_length, sTest_mahal)
        movingavg_mahal = pd.Series(sTest_mahal).rolling(windows_length).mean().to_numpy()
        
    else:
        sTest_mahal = None

    # Plotting
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Add a general title
    fig.suptitle(str(Group_Anomaly.index[idx]), fontsize=16)
    
    # RRCF
    axs[0, 0].plot(score_rrcf, label='Original')
    axs[0, 0].plot(smoothed_rrcf, label='Smoothed filt', color='red', linewidth=2)
    axs[0, 0].plot(movingavg_rrcf, label='Smoothed avg', color='purple', linewidth=2)
    axs[0, 0].fill_between(range(train_start, train_end), min(score_rrcf), max(score_rrcf), facecolor='green', alpha=0.25)
    axs[0, 0].set_title('Anomaly Score with RRCF')
    axs[0, 0].set_xlabel('Ore di lavoro [h]')
    axs[0, 0].set_ylabel('SCORE')
    axs[0, 0].legend()

    # IF
    axs[0, 1].plot(score_if, label='Original')
    axs[0, 1].plot(smoothed_if, label='Smoothed filt', color='red', linewidth=2)
    axs[0, 1].plot(movingavg_if, label='Smoothed avg', color='purple', linewidth=2)
    axs[0, 1].fill_between(range(train_start, train_end), min(score_if), max(score_if), facecolor='green', alpha=0.25)
    axs[0, 1].set_title('Anomaly Score with IF')
    axs[0, 1].set_xlabel('Ore di lavoro [h]')
    axs[0, 1].set_ylabel('SCORE')
    axs[0, 1].legend()

    # LOF
    axs[0, 2].plot(score_lof, label='Original')
    axs[0, 2].plot(smoothed_lof, label='Smoothed filt', color='red', linewidth=2)
    axs[0, 2].plot(movingavg_lof, label='Smoothed avg', color='purple', linewidth=2)
    axs[0, 2].fill_between(range(train_start, train_end), min(score_lof), max(score_lof), facecolor='green', alpha=0.25)
    axs[0, 2].set_title('Anomaly Score with LOF')
    axs[0, 2].set_xlabel('Ore di lavoro [h]')
    axs[0, 2].set_ylabel('SCORE')
    axs[0, 2].legend()

    # OCSVM
    axs[1, 0].plot(score_ocsvm, label='Original')
    axs[1, 0].plot(smoothed_ocsvm, label='Smoothed filt', color='red', linewidth=2)
    axs[1, 0].plot(movingavg_ocsvm, label='Smoothed avg', color='purple', linewidth=2)
    axs[1, 0].fill_between(range(train_start, train_end), min(score_ocsvm), max(score_ocsvm), facecolor='green', alpha=0.25)
    axs[1, 0].set_title('Anomaly Score with OCSVM')
    axs[1, 0].set_xlabel('Ore di lavoro [h]')
    axs[1, 0].set_ylabel('SCORE')
    axs[1, 0].legend()

    # MAHAL
    if sTest_mahal is not None:
        axs[1, 1].plot(sTest_mahal, label='Original')
        axs[1, 1].plot(smoothed_mahal, label='Smoothed filt', color='red', linewidth=2)
        axs[1, 1].plot(movingavg_mahal, label='Smoothed avg', color='purple', linewidth=2)
        axs[1, 1].fill_between(range(train_start, train_end), min(sTest_mahal), max(sTest_mahal), facecolor='green', alpha=0.25)
        axs[1, 1].set_title('Anomaly Score with MD')
        axs[1, 1].set_xlabel('Ore di lavoro [h]')
        axs[1, 1].set_ylabel('SCORE')
        axs[1, 1].legend()
    else:
        axs[1, 1].text(0.5, 0.5, 'Matrix is not positive definite.\nCannot compute robust covariance.', 
                    horizontalalignment='center', verticalalignment='center', transform=axs[1, 1].transAxes)

    # Comparison
    axs[1, 2].plot(MinMaxScaler().fit_transform(filtfilt(np.ones(windows_length), windows_length, movingavg_rrcf).reshape(-1, 1)), label='RRCF', linewidth=3)
    axs[1, 2].plot(MinMaxScaler().fit_transform(filtfilt(np.ones(windows_length), windows_length, movingavg_if).reshape(-1, 1)), label='IF', linewidth=3)
    axs[1, 2].plot(MinMaxScaler().fit_transform(filtfilt(np.ones(windows_length), windows_length, movingavg_lof).reshape(-1, 1)), label='LOF', linewidth=3)
    axs[1, 2].plot(MinMaxScaler().fit_transform(filtfilt(np.ones(windows_length), windows_length, movingavg_ocsvm).reshape(-1, 1)), label='OCSVM', linewidth=3)
    if sTest_mahal is not None:
        axs[1, 2].plot(MinMaxScaler().fit_transform(filtfilt(np.ones(windows_length), windows_length, movingavg_mahal).reshape(-1, 1)), label='MD', linewidth=3)
    axs[1, 2].fill_between(range(train_start, train_end), 0, 1, facecolor='green', alpha=0.25)
    axs[1, 2].set_title('Comparison between all the methods')
    axs[1, 2].set_xlabel('Ore di lavoro [h]')
    axs[1, 2].set_ylabel('Normalized Score')
    axs[1, 2].legend()

    # Define the x-tick locations and labels
    x_ticks = np.arange(0, len(data_test), 1000)  # Adjust the step size as needed
    x_labels = x_ticks / 2  # Divide by 2 to convert to half-hour increments

    # Update the x-ticks and labels for each subplot
    for ax in axs.flat:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

    fig.tight_layout()
    plt.show()