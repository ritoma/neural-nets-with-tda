import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import kmapper as km
import tensorflow.compat.v2 as tf
from pllay import to_tf_dataset

from mnist_eval import preprocess, MNIST_CNN_PLLay

def scaling(data, feature_range = (0, 1)):
	'''
	Scales the data to the range of feature_range
	Input: data, feature_range
	Output: scaled data
	'''
	scaler = MinMaxScaler(feature_range = feature_range)
	data = scaler.fit_transform(data)
	return data

def mapper_pca_vne(data, layer, resolution, gain):
	'''
	Creates a mapper object for the given data with PCA projection and VNE
	Input: data, layer, resolution, gain
	Output: html visualisation
	'''

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

	html = mapper.visualize(
        graph,
		path_html = 'kepler-mapper-output_layer-' + str(layer) + '_' + str(resolution) + '_' + str(gain) + '.html',
	)

	return html

if __name__ == '__main__':

	x_processed_file_list, y_file, model_cnn_file_array, model_cnn_pllay_file_array, model_cnn_pllay_input_file_array = preprocess()

	(x_train_processed, x_test_processed) = np.load(
				x_processed_file_list[0], allow_pickle=True)
	(y_train, y_test) = np.load(y_file, allow_pickle=True)

	test_dataset = to_tf_dataset(x=x_test_processed, y=y_test,
				batch_size=16)

	model_cnn_pllay = MNIST_CNN_PLLay()
	model_cnn_pllay.compile(optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['sparse_categorical_accuracy'])
	model_cnn_pllay.load_weights(
		model_cnn_pllay_file_array[0][0])
	output = model_cnn_pllay.predict(test_dataset)

	first_layer_weights = model_cnn_pllay.layers[0].get_weights()[0]
	first_layer_biases  = model_cnn_pllay.layers[0].get_weights()[1]
	second_layer_weights = model_cnn_pllay.layers[1].get_weights()[0]
	second_layer_biases  = model_cnn_pllay.layers[1].get_weights()[1]

	print(first_layer_weights.shape)
	print(second_layer_weights.shape)

	def reshape_dimensions(weights):
		nsamples, nx, ny, n1 = weights.shape
		weights = weights.reshape((nsamples*nx*ny, n1))
		return weights

	# nsamples, nx, ny, n1 = second_layer_weights.shape
	# second_layer_weights = second_layer_weights.reshape((nsamples*nx*ny, n1))

	first_layer_weights_reshaped = reshape_dimensions(first_layer_weights)
	second_layer_weights_reshaped = reshape_dimensions(second_layer_weights)

	weights = [first_layer_weights_reshaped, second_layer_weights_reshaped]
	resolution = 10
	gains = [2, 3, 4]

	for weight in weights:
		for gain in gains:
			try:
				mapper_pca_vne(weight, (weights.index(weight) + 1), resolution, gain)
			except Exception as e:
				print(e)