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

game_result = dataset1_csv.values[:,3] - dataset1_csv.values[:,20]
basketball = {
    "data": np.delete(dataset1_csv.values, [0,2,3,19,20], axis=1),
    "target": np.vectorize(lambda x : 0 if x < 1 else 1)(game_result)
}
encoder=LabelEncoder()
for idx in range(len(basketball["data"][0])):
    if type(basketball["data"][0][idx]) is str:
        basketball["data"][:,idx]=encoder.fit_transform(basketball["data"][:,idx])

basketball["data"] = StandardScaler().fit_transform(basketball["data"])
    
digits = load_digits()
digit = {
    "data": StandardScaler().fit_transform(digits.data),
    "target": digits.target
}

from sklearn.metrics import adjusted_mutual_info_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM

def cluster_acc(Y,clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
    return accuracy_score(Y,pred)

kValues = [2, 4, 6, 8, 10, 12, 14, 20, 40, 60, 80, 100]

def runKMeansClustering(data, target, filePrefix, printCenters=False):
    print(filePrefix + " KMeans")
    results = {
        "accuracy": [],
        "ami": [],
        "score": []
    }
    
    for k in kValues:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        accuracy = cluster_acc(target, kmeans.predict(data))
        ami = adjusted_mutual_info_score(target, kmeans.predict(data))
        results["accuracy"].append(accuracy)
        results["ami"].append(ami)
        results["score"].append(kmeans.score(data))
        if printCenters:
            tmp = pd.DataFrame(np.std(kmeans.cluster_centers_, axis=0))
            tmp.to_csv(str(k)+filePrefix+'_kmeans_centers.csv')
            
    tmp = pd.DataFrame(list(zip(kValues, results["accuracy"])))
    tmp.to_csv(filePrefix + '_kmeans_accuracy.csv')
    
    tmp = pd.DataFrame(list(zip(kValues, results["ami"])))
    tmp.to_csv(filePrefix + '_kmeans_ami.csv')
    
    tmp = pd.DataFrame(list(zip(kValues, results["score"])))
    tmp.to_csv(filePrefix + '_kmeans_log_likelihood.csv')
    
def runGMMClustering(data, target, filePrefix, printCenters=False):
    print(filePrefix + " GMM")
    results = {
        "accuracy": [],
        "ami": [],
        "score": []
    }
    
    for k in kValues:
        gmm = GMM(n_components=k)
        gmm.fit(data)
        accuracy = cluster_acc(target, gmm.predict(data))
        ami = adjusted_mutual_info_score(target, gmm.predict(data))
        results["accuracy"].append(accuracy)
        results["ami"].append(ami)
        results["score"].append(gmm.score(data))
        if printCenters:
            tmp = pd.DataFrame(np.std(gmm.means_, axis=0))
            tmp.to_csv(str(k)+filePrefix+'_gmm_centers.csv')
            
    tmp = pd.DataFrame(list(zip(kValues, results["accuracy"])))
    tmp.to_csv(filePrefix + '_gmm_accuracy.csv')
    
    tmp = pd.DataFrame(list(zip(kValues, results["ami"])))
    tmp.to_csv(filePrefix + '_gmm_ami.csv')
    
    tmp = pd.DataFrame(list(zip(kValues, results["score"])))
    tmp.to_csv(filePrefix + '_gmm_sse.csv')

runKMeansClustering(basketball["data"], basketball["target"], "basketball", printCenters=True)
runGMMClustering(basketball["data"], basketball["target"], "basketball", printCenters=True)
runKMeansClustering(digit["data"], digit["target"], "digit", printCenters=True)
runGMMClustering(digit["data"], digit["target"], "digit", printCenters=True)

from sklearn.metrics.pairwise import pairwise_distances
def reconstructionError(transformer,data):
    X_train = transformer.transform(data)
    X_projected = transformer.inverse_transform(X_train)
    return ((data - X_projected) ** 2).mean()

def reconstructionErrorRP(projections,X):
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p@W)@(X.T)).T # Unproject projected data
    errors = np.square(X-reconstructed)
    return np.nanmean(errors)

def pairwiseDistCorr(X1,X2):
    assert X1.shape[0] == X2.shape[0]
    
    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)
    return np.corrcoef(d1.ravel(),d2.ravel())[0,1]

from sklearn.decomposition import PCA

reconstructionErrors = []
explainedVariances = []
dims_bb = [2,4,10,20,30,35]
for dim in dims_bb:
    pca = PCA(n_components=dim)
    pca.fit(basketball["data"])
    
    explainedVariances.append(pca.explained_variance_)
    reconstruction = reconstructionError(pca, basketball["data"])
    reconstructionErrors.append(reconstruction)
    
    pcaBasketballData = pca.fit_transform(basketball["data"])
    runKMeansClustering(pcaBasketballData, basketball["target"], "pca_basketball_" + str(dim))
    runGMMClustering(pcaBasketballData, basketball["target"], "pca_basketball_" + str(dim))

tmp = pd.DataFrame(list(zip(dims_bb, explainedVariances)))
tmp.to_csv('pca_basketball_explained_variances.csv')

tmp = pd.DataFrame(list(zip(dims_bb, reconstructionErrors)))
tmp.to_csv('pca_basketball_reconstruction_error.csv')

reconstructionErrors = []
explainedVariances = []
dims_digit = [2,4,10,20,30,40,50,60]
for dim in dims_digit:
    pca = PCA(n_components=dim)
    pca.fit(digit["data"])
    
    explainedVariances.append(pca.explained_variance_)
    reconstruction = reconstructionError(pca, digit["data"])
    reconstructionErrors.append(reconstruction)
    
    pcadigitData = pca.fit_transform(digit["data"])
    runKMeansClustering(pcadigitData, digit["target"], "pca_digit_" + str(dim))
    runGMMClustering(pcadigitData, digit["target"], "pca_digit_" + str(dim))

tmp = pd.DataFrame(list(zip(dims_digit, explainedVariances)))
tmp.to_csv('pca_digit_explained_variances.csv')

tmp = pd.DataFrame(list(zip(dims_digit, reconstructionErrors)))
tmp.to_csv('pca_digit_reconstruction_error.csv')

from sklearn.decomposition import FastICA

reconstructionErrors = []
kurtosis = []
dims_bb = [2,4,10,20,30,35]
for dim in dims_bb:
    ica = FastICA(n_components=dim)
    ica.fit(basketball["data"])
    
    reconstruction = reconstructionError(ica, basketball["data"])
    reconstructionErrors.append(reconstruction)
    
    icaBasketballData = ica.fit_transform(basketball["data"])
    
    tmp = pd.DataFrame(icaBasketballData)
    tmp = tmp.kurt(axis=0)
    kurtosis.append(tmp.abs().mean())
    
    runKMeansClustering(icaBasketballData, basketball["target"], "ica_basketball_" + str(dim))
    runGMMClustering(icaBasketballData, basketball["target"], "ica_basketball_" + str(dim))

tmp = pd.DataFrame(list(zip(dims_bb, kurtosis)))
tmp.to_csv('ica_basketball_kurtosis.csv')

tmp = pd.DataFrame(list(zip(dims_bb, reconstructionErrors)))
tmp.to_csv('ica_basketball_reconstruction_error.csv')

reconstructionErrors = []
kurtosis = []
dims_digit = [2,4,10,20,30,40,50,60]
for dim in dims_digit:
    ica = FastICA(n_components=dim)
    ica.fit(digit["data"])
    
    reconstruction = reconstructionError(ica, digit["data"])
    reconstructionErrors.append(reconstruction)
    
    icadigitData = ica.fit_transform(digit["data"])
    
    tmp = pd.DataFrame(icadigitData)
    tmp = tmp.kurt(axis=0)
    kurtosis.append(tmp.abs().mean())
    
    runKMeansClustering(icadigitData, digit["target"], "ica_digit_" + str(dim))
    runGMMClustering(icadigitData, digit["target"], "ica_digit_" + str(dim))

tmp = pd.DataFrame(list(zip(dims_digit, kurtosis)))
tmp.to_csv('ica_digit_kurtosis.csv')

tmp = pd.DataFrame(list(zip(dims_digit, reconstructionErrors)))
tmp.to_csv('ica_digit_reconstruction_error.csv')

from sklearn.random_projection import SparseRandomProjection
from itertools import product

# # reconstructionErrors = defaultdict(list)
# pairwise_distance_correlation = defaultdict(list)
# dims_bb = [2,4,10,20,30,35]
# for i in range(10):
#     for dim in dims_bb:
#         rp = SparseRandomProjection(random_state=i, n_components=dim)
#         rp.fit(basketball["data"])

#         rpBasketballData = rp.fit_transform(basketball["data"])

#         pairwise_distance_correlation[dim].append(pairwiseDistCorr(rpBasketballData, basketball["data"]))

#         runKMeansClustering(rpBasketballData, basketball["target"], "rp_basketball_" + str(dim) + "_" + str(i))
#         runGMMClustering(rpBasketballData, basketball["target"], "rp_basketball_" + str(dim) + "_" + str(i))

# tmp = pd.DataFrame(pairwise_distance_correlation)
# tmp.to_csv('rp_basketball_pairwise_dist_corr.csv')

# tmp = pd.DataFrame(reconstructionErrors)
# tmp.to_csv('rp_basketball_reconstruction_error.csv')

# reconstructionErrors = defaultdict(list)
pairwise_distance_correlation = defaultdict(list)
dims_digit = [2,4,10,20,30,40,50,60]
for i in range(10):
    for dim in dims_digit:
        rp = SparseRandomProjection(random_state=i, n_components=dim)
        rp.fit(digit["data"])

    #     reconstruction = reconstructionErrorRP(rp, digit["data"])
    #     reconstructionErrors[i].append(reconstruction)

        rpDigitData = rp.fit_transform(digit["data"])

        pairwise_distance_correlation[dim].append(pairwiseDistCorr(rpDigitData, digit["data"]))

#         runKMeansClustering(rpDigitData, digit["target"], "rp_digit_" + str(dim) + "_" + str(i))
#         runGMMClustering(rpDigitData, digit["target"], "rp_digit_" + str(dim) + "_" + str(i))

tmp = pd.DataFrame(pairwise_distance_correlation)
tmp.to_csv('rp_digit_pairwise_dist_corr.csv')

# tmp = pd.DataFrame(reconstructionErrors)
# tmp.to_csv('rp_digit_reconstruction_error.csv')

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

        
rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',n_jobs=7)
fs_bball = rfc.fit(basketball["data"], basketball["target"]).feature_importances_ 
rfcBasketballData = ImportanceSelect(rfc, 10).fit_transform(basketball["data"], basketball["target"])

tmp = pd.Series(fs_bball)
tmp.to_csv("rfc_basketball_feature_importance.csv")

runKMeansClustering(rfcBasketballData, basketball["target"], "rfc_basketball_")
runGMMClustering(rfcBasketballData, basketball["target"], "rfc_basketball_")


rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',n_jobs=7)
fs_digit = rfc.fit(digit["data"], digit["target"]).feature_importances_ 
for i in [10, 15, 20, 25, 30]:    
    rfcDigitData = ImportanceSelect(rfc, i).fit_transform(digit["data"], digit["target"])

#     tmp = pd.Series(fs_digit)
#     tmp.to_csv("rfc_digit_feature_importance.csv")

    runKMeansClustering(rfcDigitData, digit["target"], "rfc_digit_" + str(i) + "_")
    runGMMClustering(rfcDigitData, digit["target"], "rfc_digit_" + str(i) + "_")