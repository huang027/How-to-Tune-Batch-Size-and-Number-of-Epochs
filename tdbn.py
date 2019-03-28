import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np

def create_model():
    model=Sequential()
    model.add(Dense(12,input_dim=8,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
if __name__=='__main__':
    dataset=pd.read_csv('G:\SQL_CJ\pima-indians-diabetes.csv',header=None)
    seed = 7
    np.random.seed(seed)
    X = dataset.ix[:, 0:7]
    Y = dataset.ix[:, 8]
    model = KerasClassifier(build_fn=create_model, verbose=0)
    batch_size = [10, 20, 30, 40, 60, 80, 100]
    epochs = [10, 50, 100]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X, Y)
    print("Best:%f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, std, param in zip(means, stds, params):
        print("%f(%f)with:%r" % (mean, std, param))