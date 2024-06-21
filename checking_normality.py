import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Load data
filename = "filepath/HR_data.csv"
df = pd.read_csv(filename)

# Preprocess data
unique_individuals = df['Individual'].unique()
train_inds, test_inds = train_test_split(unique_individuals, test_size=0.3, random_state=42)

train_df = df[df['Individual'].isin(train_inds)]
test_df = df[df['Individual'].isin(test_inds)]

sc = MinMaxScaler(feature_range=(0,1))
X_train = sc.fit_transform(train_df.iloc[:, 0:6])
y_train = train_df["Frustrated"].values
X_test = sc.transform(test_df.iloc[:, 0:6])
y_test = test_df["Frustrated"].values

# ANN 
n_runs = 100
ANN_accuracies = []

for i in range(n_runs):
    ANN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5), random_state=i)
    ANN.fit(X_train, y_train)
    ANN_accuracies.append(ANN.score(X_test, y_test))

# Random Forest
RF_accuracies = []
estimators = 50

for i in range(n_runs):
    RF = RandomForestClassifier(random_state=i, n_estimators=estimators)
    RF.fit(X_train, y_train)
    RF_accuracies.append(RF.score(X_test, y_test))

# Bootstrapping
n_bootstraps = 1000
bootstrap_means_ANN = [np.mean(np.random.choice(ANN_accuracies, size=len(ANN_accuracies), replace=True)) for _ in range(n_bootstraps)]
bootstrap_means_RF = [np.mean(np.random.choice(RF_accuracies, size=len(RF_accuracies), replace=True)) for _ in range(n_bootstraps)]

# Plotting
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.hist(bootstrap_means_ANN, bins=30, alpha=0.75, color='blue', edgecolor='black')
plt.title('Bootstrap Distribution of ANN Mean Accuracies')
plt.xlabel('Mean Accuracy')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(bootstrap_means_RF, bins=30, alpha=0.75, color='green', edgecolor='black')
plt.title('Bootstrap Distribution of RF Mean Accuracies')
plt.xlabel('Mean Accuracy')

plt.tight_layout()
plt.show()

