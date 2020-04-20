import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Activation, Dropout
from tensorflow.keras.optimizers import Adam

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

print(data.describe())

lane_curve = data['median_lane_curve'].to_numpy()
center_angle = data['median_center_angle'].to_numpy()
vehicle_offset = data['median_vehicle_offset'].to_numpy()

lane_curve /= 5000
center_angle = center_angle * scipy.pi / 180

Y = data['steering_angle'].to_numpy()

Y = Y * scipy.pi / 180
print('Min Max: ',np.min(Y), np.max(Y))

sample_size = 500000

#X = center_angle[:10000].reshape(-1,1)
X = np.concatenate([lane_curve[:sample_size].reshape(-1,1), vehicle_offset[:sample_size].reshape(-1,1), center_angle[:sample_size].reshape(-1,1)], axis=1)
print(X.shape)
Y = Y[:sample_size].reshape(-1,1)
print(Y.shape)

data = np.concatenate((Y, X), axis=1)

print('\nBallancing Data')
print('Starting Size: ', len(data))
left_samples = []
right_samples = []
center_samples = []
for sample in data:
    if sample[0] < 0:
        left_samples.append(sample)
    elif sample[0] > 0:
        right_samples.append(sample)
    else:
        center_samples.append(sample)
print('Before Balancing')
print('Left: ',len(left_samples))
print('Right: ',len(right_samples))
print('Straight: ',len(center_samples))

center_samples = center_samples[:len(center_samples)//8]

left_samples = np.array(left_samples)
right_samples = np.array(right_samples)
center_samples = np.array(center_samples)

balanced_data = []
if len(left_samples) > 0:
    balanced_data.extend(list(left_samples))

if len(right_samples) > 0:
    balanced_data.extend(list(right_samples))

if len(center_samples) > 0:
    balanced_data.extend(list(center_samples))

print('After Balancing')
print('Left: ',len(left_samples))
print('Right: ',len(right_samples))
print('Straight: ',len(center_samples))

print('Ending Size: ',len(balanced_data))
balanced_data = np.array(balanced_data)


Y = balanced_data[:,0]
X = balanced_data[:,1:]

print(X[0], Y[0])

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print('Train Shape: ',X_train.shape, Y_train.shape)
print(X_train[0])

model = Sequential()
model.add(Dense(128, input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('tanh'))

model.compile(loss='mse', optimizer=Adam(learning_rate=1e-4))
print(model.summary())
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=16, epochs=30)

model.save('dnn.h5')

pred = model.predict(X_test)
plt.plot([i for i in range(100)], Y_test[:100])
plt.plot([i for i in range(100)], pred[:100])
plt.show()