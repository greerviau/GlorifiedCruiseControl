import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy

def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def remove_outliers(x):
    mean = np.mean(x)
    standard_deviation = np.std(x)
    distance_from_mean = abs(x - mean)
    max_deviations = 2
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    return x[not_outlier]


files = os.listdir('data_processed')
data = []

for f in files:
    path_to_csv = os.path.join('data_processed', f)
    csv = os.listdir(path_to_csv)[0]
    csv_final_path = os.path.join(path_to_csv, csv)
    print('Loading ',csv_final_path)
    d = pd.read_csv(csv_final_path)[100:]
    #print(d.head())
    data.append(d)

data = pd.concat(data, axis=0, ignore_index=True)
'''
data = pd.read_csv('data_processed/sess_07_processed/sess_07.csv')
'''
print(data.describe())

lane_curve = data['median_lane_curve'].to_numpy()
center_angle = data['median_center_angle'].to_numpy()
vehicle_offset = data['median_vehicle_offset'].to_numpy()

Y = data['steering_angle'].to_numpy()

'''
lane_curve = normalize(lane_curve)
center_angle = normalize(center_angle)
vehicle_offset = normalize(vehicle_offset)
'''

print(np.max(lane_curve))
lane_curve = lane_curve / 5000
center_angle = center_angle * scipy.pi / 180

Y = Y * scipy.pi / 180
print('Min Max: ',np.min(Y), np.max(Y))

sample_size = 500000

#X = center_angle[:10000].reshape(-1,1)
X = np.concatenate([lane_curve[:sample_size].reshape(-1,1), vehicle_offset[:sample_size].reshape(-1,1), center_angle[:sample_size].reshape(-1,1)], axis=1)

Y = Y[:sample_size].reshape(-1,1)
print(Y.shape)

#X = np.concatenate([X[1:,:], Y[:-1,:]], axis=1)

#Y = Y[1:,:]
'''
for i in range(100):
    print(X[i], Y[i])
print(X.shape)

plot_data = np.concatenate([X[:1000,:], Y[:1000]], axis=1)
plt.plot([i for i in range(len(plot_data))], plot_data)
plt.show()
'''

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
from sklearn.metrics import r2_score, mean_squared_error

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print('Train Shape: ',X_train.shape, Y_train.shape)
print(X_train[0])

reg = sm.OLS(Y_train, X_train).fit()

print(reg.summary())

y_train_pred = reg.predict(X_train)

plot_samples = 100

print('Train R2 Score: ',r2_score(Y_train, y_train_pred))
print('Train MSE Score: ',mean_squared_error(Y_train, y_train_pred))

plt.plot([i for i in range(plot_samples)], Y_train[:plot_samples])
plt.plot([i for i in range(plot_samples)], y_train_pred[:plot_samples])
plt.show()

y_test_pred = reg.predict(X_test)

print('Test R2 Score: ',r2_score(Y_test, y_test_pred))
print('Test MSE Score: ',mean_squared_error(Y_test, y_test_pred))

plt.plot([i for i in range(plot_samples)], Y_test[:plot_samples])
plt.plot([i for i in range(plot_samples)], y_test_pred[:plot_samples])
plt.show()


print(Y_test[:10])
print(y_test_pred[:10].reshape(-1,1))


import pickle as pkl 

pkl.dump(reg, open('regression_model.pkl', 'wb'))

'''
prev_steering_angle = X[0][3]
print('P Steer ',prev_steering_angle)
steering_preds = []
steering_preds_wo = []
for i in range(100):
    test_x = np.copy(X[i])
    print(test_x)
    test_x[3] = prev_steering_angle
    print(test_x)
    steering_pred = round(reg.predict(test_x)[0],1)
    steering_pred_wo = round(reg.predict(X[i])[0],1)
    print(steering_pred, steering_pred_wo, Y[i])
    prev_steering_angle = steering_pred

    steering_preds.append(steering_pred)
    steering_preds_wo.append(steering_pred_wo)
    #print(steering_pred, Y_test[i])

plt.plot([i for i in range(len(steering_preds))], Y[:len(steering_preds)])
plt.plot([i for i in range(len(steering_preds))], steering_preds)
plt.plot([i for i in range(len(steering_preds))], steering_preds_wo)
plt.show()
'''
