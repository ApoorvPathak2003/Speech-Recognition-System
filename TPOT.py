import pandas as pd
import tpot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

mean_normal_feature_dataframe = pd.read_csv('path_to_csv_file')

label_encoder = LabelEncoder()
mean_normal_feature_dataframe['Labels'] = label_encoder.fit_transform(mean_normal_feature_dataframe['Labels'])

X = mean_normal_feature_dataframe.drop(columns = ['Labels'])
Y_label = mean_normal_feature_dataframe['Labels']

x_train, x_test, y_train, y_test = train_test_split(X, Y_label, random_state = 42, shuffle = True, train_size = 0.8, test_size = 0.2)

tpot_model = tpot.TPOTClassifier(verbosity = 3,
                      scoring = "balanced_accuracy", 
                      random_state = 42,
                      periodic_checkpoint_folder = "tpot.txt", 
                      n_jobs = -1,
                      generations = 10,
                      population_size = 100)

tpot_model.fit(x_test, y_test)

print('Fitted Pipeline: ', tpot_model.fitted_pipeline_)

print('Score: ', tpot_model.score(x_test, y_test))

tpot_model.export('best_model.py')
