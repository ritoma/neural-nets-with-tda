import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import kmapper as km

def scaling(data, feature_range = (0, 1)):
	'''
	Scales the data to the range of feature_range
	Input: data, feature_range
	Output: scaled data
	'''
	scaler = MinMaxScaler(feature_range = feature_range)
	data = scaler.fit_transform(data)
	return data

def mapper_pca_vne(data, resolution, gain):

	# data = scaling(data)

	# Initialise mapper
	mapper = km.KeplerMapper(verbose = 2)

	# Fit and transform data
	# Data is projected onto its two principal components
	projected_data = mapper.project(
		data,
		projection = sklearn.decomposition.KernelPCA(n_components = 2),
		distance_matrix = 'seuclidean'
	)
	projected_data = mapper.fit_transform(data, projection = sklearn.decomposition.KernelPCA(n_components = 2))

	# Cluster data
	# Single linkage clustering is used with the Variance Normalised Euclidean metric
	graph = mapper.map(
		projected_data,
		clusterer = sklearn.cluster.AgglomerativeClustering(linkage = 'single', affinity = 'euclidean'),
		cover = km.Cover(n_cubes = resolution, perc_overlap = (1 - (1/gain))),
	)

	html = mapper.visualize(graph,
		path_html = 'kepler-mapper-output_' + str(resolution) + '_' + str(gain) + '.html',
	)