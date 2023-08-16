# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display
from tqdm import tqdm

"""# **Data Loading, cleaning**"""

audio_dataset_path='/content/drive/MyDrive/EDA/recordings/recordings'
metadata = pd.read_csv('/content/drive/MyDrive/EDA/speakers_all.csv')
metadata.head()

metadata.drop(metadata.columns[9:12],axis = 1, inplace = True)
print(metadata.columns)
metadata.describe()

metadata.groupby("native_language")['age'].describe().sort_values(by=['count'],ascending=False)

# file_missing
metadata.groupby("file_missing?")['age'].describe().sort_values(by=['count'],ascending=False)

# Count the total audio files given
print (len([name for name in os.listdir('/content/drive/MyDrive/EDA/recordings/recordings') if os.path.isfile(os.path.join('/content/drive/MyDrive/EDA/recordings/recordings', name))]))

# filename column. This time we just print out the first 10 records.
metadata.groupby("filename")['age'].describe().sort_values(by=['count'],ascending=False).head(10)

# Cross-tab. Again, just print the first 10 record
metadata.groupby("filename")['file_missing?'].describe().sort_values(by=['count'],ascending=False).head(10)
# pd.crosstab(df['filename'],df['file_missing?']) as an alternative method

#finally droping the file missing == True


metadata.drop(metadata.index[0:32],axis = 0, inplace = True)
metadata.reset_index(inplace = True)

metadata[metadata.filename == 'nicaragua']
metadata.drop(metadata.index[1512],axis = 0, inplace = True)
metadata[metadata.filename == 'nicaragua']
#metadata.head()

metadata.reset_index(inplace = True)

len(metadata)

l = [name for name in os.listdir('/content/drive/MyDrive/EDA/recordings/recordings') if os.path.isfile(os.path.join('/content/drive/MyDrive/EDA/recordings/recordings', name))]
n = []
for i in l:
    n.append(i.removesuffix('.mp3'))
p = metadata['filename'].isin(n)
for i in range(len(metadata)):
    if p[i] == False:
        print(metadata.iloc[i])
        print(i)

metadata.drop(metadata.index[1738],axis = 0, inplace = True)
metadata.reset_index(inplace = True, drop = True)
#metadata.iloc[1738]

len(metadata)

metadata.groupby("sex")['age'].describe()

metadata[metadata['native_language']=='english']

metadata['accent'] = metadata['native_language'].apply(lambda x: 'native' if x=='english' else 'non-native')

metadata

"""# **Feature Extraction**"""

def features_extractor(file):
    audio, sample_rate = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc= 40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name = '/content/drive/MyDrive/EDA/recordings/recordings/' + str(l[index_num])
    final_class_labels=row["accent"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])

extracted_features_df = pd.DataFrame(extracted_features,columns=['feature','class'])

extracted_features_df['class'] = extracted_features_df['class'].apply(lambda x: 1 if x =='native' else 0 )
extracted_features_df.head()

"""# **Model function**"""

# import necessary libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def model (X_train,X_test,y_train,y_test):
# create a list of classifiers to train
    classifiers = [
        ('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors = 11)),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Random Forest', RandomForestClassifier(n_estimators = 500)),
        ('LogisticRegression', LogisticRegression())
    ]

    # train each classifier and print its accuracy score
    for name, clf in classifiers:
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        report = classification_report(y_test, y_pred)
        print(f'{name} accuracy: {accuracy:.2f} f1 score: {f1:.3f} ')
        print(f'{name} \n {report}')

"""# **Without handling Imbalanced Dataset**


> X_train, X_test, y_train, y_test


"""

X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)

model (X_train,X_test,y_train,y_test)

"""# **UNDER SAMPLING**

> X_train_u, X_test_u, y_train_u, y_test_u


"""

# Class count
count_class_0, count_class_1 = extracted_features_df['class'].value_counts()
count_class_0, count_class_1

# Divide by class
df_class_under_0 = extracted_features_df[extracted_features_df['class'] == 0]
df_class_under_1 = extracted_features_df[extracted_features_df['class'] == 1]

# Undersample 0-class and concat the DataFrames of both class
df_class_0_under = df_class_under_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_under_1], axis=0)

print('Random under-sampling:')
print(df_test_under['class'].value_counts())

### Split the dataset into independent and dependent dataset
X=np.array(df_test_under['feature'].tolist())
y=np.array(df_test_under['class'].tolist())

from sklearn.model_selection import train_test_split
X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)

model (X_train_u,X_test_u,y_train_u,y_test_u)



"""# **Over Sampling**


> X_train_o, X_test_o, y_train_o, y_test_o


"""

from imblearn.over_sampling import RandomOverSampler
# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy='minority')

X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())
X_over, y_over = oversample.fit_resample(X, y)

from sklearn.model_selection import train_test_split
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_over, y_over, test_size=0.2, random_state=15, stratify = y_over)

model (X_train_o, X_test_o, y_train_o, y_test_o)

"""# **SMOTE**



> X_train_s, X_test_s, y_train_s, y_test_s




"""

### Split the dataset into independent and dependent dataset
X_SMOTE = extracted_features_df['feature'].tolist()
y_SMOTE = extracted_features_df['class'].tolist()

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X_SMOTE, y_SMOTE)

from sklearn.model_selection import train_test_split
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sm, y_sm, test_size=0.2, random_state=15, stratify=y_sm)

model (X_train_s, X_test_s, y_train_s, y_test_s)

"""0       0.86      0.96      0.91       312
0       0.81      0.88      0.85       312
1       0.95      0.85      0.90       312
1       0.87      0.79      0.83       312
"""



"""# **HyperParameter Tunning**"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score ,precision_score

model_params = {

    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
                    'n_estimators': [200, 400,500, 600, 800]
          }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'random_state': [1,5,10]
        }
    },
    'DesicianTreeClassifier':{
        'model': DecisionTreeClassifier(),
        'params' : {
                      'max_depth': [2, 3, 5, 10, 20],
                      'min_samples_leaf': [5, 10, 20, 50, 100],
                      'criterion': ["gini", "entropy"]
}
    },
    'KNeighborsClassifier':{
               'model': KNeighborsClassifier(),
               'params':{
                  'n_neighbors' : [5,7,10,15,19,21],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}
    }
}

def para_tunnig(X_train,X_test,y_train,y_test, model_params):
    scores = []
    for model_name, mp in model_params.items():
        clf =  GridSearchCV(mp['model'], mp['params'], cv=5 ,return_train_score=False)
        clf.fit(X_train, y_train)
        y_pred = clf.best_estimator_.predict(X_test)
        ro = roc_auc_score(y_test,y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        pre  = precision_score(y_test, y_pred, average='macro')
        scores.append({
            'model': model_name,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_,
            'f1_score': f1,
            'roc_score': ro,
            'precision_score' : pre
        })
    return scores

scores =  para_tunnig(X_train,X_test,y_train,y_test, model_params)
df = pd.DataFrame(scores,columns=['model','best_score','best_params','f1_score','roc_score', 'precision_score'])
df

scores =  para_tunnig(X_train_u,X_test_u,y_train_u,y_test_u, model_params)
df = pd.DataFrame(scores,columns=['model','best_score','best_params','f1_score','roc_score', 'precision_score'])
df

scores =  para_tunnig(X_train_o,X_test_o,y_train_o,y_test_o, model_params)
df = pd.DataFrame(scores,columns=['model','best_score','best_params','f1_score','roc_score', 'precision_score'])
df

scores =  para_tunnig(X_train_s,X_test_s,y_train_s,y_test_s, model_params)
df = pd.DataFrame(scores,columns=['model','best_score','best_params','f1_score','roc_score', 'precision_score'])
df







"""# **Model check **"""

t = RandomForestClassifier(n_estimators = 800)
 t.fit(X_train_o,y_train_o)

from sklearn.metrics import classification_report, confusion_matrix
y_pred_o = t.predict(X_test_o)
report = classification_report(y_test_o, y_pred_o)

print(report)

audio_file_path='/content/drive/MyDrive/EDA/recordings/recordings/afrikaans3.mp3'
librosa_audio_data,librosa_sample_rate=librosa.load(audio_file_path)
plt.figure(figsize=(12, 4))
librosa.display.waveshow(librosa_audio_data, sr=librosa_sample_rate)
plt.show()

audio_file_path='/content/drive/MyDrive/eda test sample .opus'
librosa_audio_data,librosa_sample_rate=librosa.load(audio_file_path)
mfccs = librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs.T,axis=0)

x_sa = [mfccs_scaled_features]

t.predict(x_sa)

