import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict, Counter
from sklearn.metrics import adjusted_mutual_info_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
import sys

from sklearn.preprocessing import LabelEncoder
import pandas as pd

dataset1_csv = pd.read_csv('march_madness.csv')

# March Madness 
game_result = dataset1_csv.values[:,3] - dataset1_csv.values[:,20]
basketball = {
    "data": np.delete(dataset1_csv.values, [0,3,20], axis=1),
    "target": np.vectorize(lambda x : 0 if x < 1 else 1)(game_result)
}
encoder=LabelEncoder()
for idx in range(len(basketball["data"][0])):
    if type(basketball["data"][0][idx]) is str:
        basketball["data"][:,idx]=encoder.fit_transform(basketball["data"][:,idx])

digits = load_digits()
digit = {
    "data": StandardScaler().fit_transform(digits.data),
    "target": digits.target
}

from sklearn.preprocessing import LabelEncoder
import pandas as pd

dataset1_csv = pd.read_csv('binary_basketball.csv')
initial_dataset = dataset1_csv.values

# # March Madness 
# game_result = dataset1_csv.values[:,41]

# initial_dataset = dataset1_csv.values
# updated_dataset = np.copy(dataset1_csv.values)
# basketball = {
#     "data": np.delete(updated_dataset, [41], axis=1),
#     "target": np.vectorize(lambda x : 0 if x == "Away" else 1)(game_result)
# }

from sklearn.decomposition import PCA

dims_bb = [2,5,10,15,20,25,30,35]
nn_arch= [(5,10),(10,10),(10,20),(20,20),(20,10),(5,),(10,),(20,),(30,),(30,10),(30,20),(30,30)]

grid ={'pca__n_components':dims_bb, 'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5,alpha=0.1)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(basketball["data"],basketball["target"])
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv('bball_pca_nn.csv')
print(gs.cv_results_)
print(gs.cv_results_.keys())

nn_arch= [(5,10),(10,10),(10,20),(20,20),(20,10),(5,),(10,),(20,),(30,),(30,10),(30,20),(30,30)]


pca = PCA(n_components=10)
pca.fit(basketball["data"])
pcaBasketballData = pca.fit_transform(basketball["data"])

kmeans = KMeans(n_clusters=10)
kmeans.fit(pcaBasketballData)
clusteredBasketballData = kmeans.transform(pcaBasketballData)



grid ={'NN__hidden_layer_sizes':nn_arch}      
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5,alpha=0.1)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(clusteredBasketballData,basketball["target"])
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv('bball_pca_nn_from_clusters.csv')

from sklearn.decomposition import FastICA

dims_bb = [2,5,10,15,20,25,30,35]
nn_arch= [(5,10),(10,10),(10,20),(20,20),(20,10),(5,),(10,),(20,),(30,),(30,10),(30,20),(30,30)]

grid ={'ica__n_components':dims_bb, 'NN__hidden_layer_sizes':nn_arch}
ica = FastICA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5,alpha=0.1)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(basketball["data"],basketball["target"])
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv('bball_ica_nn.csv')

nn_arch= [(5,10),(10,10),(10,20),(20,20),(20,10),(5,),(10,),(20,),(30,),(30,10),(30,20),(30,30)]


pca = FastICA(n_components=10)
pca.fit(basketball["data"])
pcaBasketballData = pca.fit_transform(basketball["data"])

kmeans = KMeans(n_clusters=10)
kmeans.fit(pcaBasketballData)
clusteredBasketballData = kmeans.transform(pcaBasketballData)



grid ={'NN__hidden_layer_sizes':nn_arch}      
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5,alpha=0.1)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(clusteredBasketballData,basketball["target"])
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv('bball_pca_nn_from_clusters.csv')

from sklearn.random_projection import SparseRandomProjection

dims_bb = [2,5,10,15,20,25,30,35]
nn_arch= [(5,10),(10,10),(10,20),(20,20),(20,10),(5,),(10,),(20,),(30,),(30,10),(30,20),(30,30)]

grid ={'rp__n_components':dims_bb, 'NN__hidden_layer_sizes':nn_arch}
rp = SparseRandomProjection(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5,alpha=0.1)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(basketball["data"],basketball["target"])
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv('bball_rp_nn.csv')

from sklearn.ensemble import RandomForestClassifier
from sklearn.base import TransformerMixin,BaseEstimator

class ImportanceSelect(BaseEstimator, TransformerMixin):
    def __init__(self, model, n=1):
         self.model = model
         self.n = n
    def fit(self, *args, **kwargs):
         self.model.fit(*args, **kwargs)
         return self
    def transform(self, X):
         return X[:,self.model.feature_importances_.argsort()[::-1][:self.n]]

dims_bb = [2,5,10,15,20,25,30,35]
nn_arch= [(5,10),(10,10),(10,20),(20,20),(20,10),(5,),(10,),(20,),(30,),(30,10),(30,20),(30,30)]

rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',n_jobs=7)
filtr = ImportanceSelect(rfc)
grid ={'filter__n':dims,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('filter',filtr),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(basketball["data"],basketball["target"])
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv('bball_rp_nn.csv')