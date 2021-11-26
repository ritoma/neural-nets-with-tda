import numpy as np
import tensorflow.compat.v2 as tf
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import kmapper as km
import sklearn
from pllay import *

tf.enable_v2_behavior()


# Global Variables
nmax_diag = 32
corrupt_prob_list = [0.1]
noise_prob_list = [0.1]
nCn = len(corrupt_prob_list)
batch_size = 16
nTimes=1


class MNIST_CNN(tf.keras.Model):
    def __init__(self, name='mnistcnn', filters=32, kernel_size=3, unitsDense=64, **kwargs):
        super(MNIST_CNN, self).__init__(name=name, **kwargs)
        self.layer1_1 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation='relu')
        self.layer1_2 = tf.keras.layers.Conv2D(1, kernel_size, padding="same", activation='relu')
        #self.layer3 = tf.keras.layers.Dense(unitsDense, activation='relu', name='dense_2') 
        #self.layer4 = tf.keras.layers.Dense(10, name='predictions')

    def call(self, x):
        xg, xl1, xl2, xd = tf.split(x, [784, 100, 162, 8*nmax_diag], axis=-1)
        xg = tf.reshape(xg, [16, 28, 28, 1])
        xg1 = self.layer1_1(xg)
        xg1 = self.layer1_2(xg1)
        #xg1 = tf.reshape(xg1, [16, 784])
        x = xg1
        #x = self.layer3(x)
        #x = self.layer4(x)
        print(x.shape)
        return x


class MNIST_CNN_PLLay_Input(tf.keras.Model):
    def __init__(self, name='mnistcnnpllayinput', filters=32, kernel_size=3, unitsDense=64, unitsTopInput=32, **kwargs):
        super(MNIST_CNN_PLLay_Input, self).__init__(name=name, **kwargs)
        self.layer1_1 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation='relu')
        self.layer1_2 = tf.keras.layers.Conv2D(1, kernel_size, padding="same", activation='relu')
        self.layer2_1 = GThetaLayer(unitsTopInput)
        self.layer2_2 = GThetaLayer(unitsTopInput)
        self.layer3 = tf.keras.layers.Dense(unitsDense, activation='relu', name='dense_2') 
        self.layer4 = tf.keras.layers.Dense(10, name='predictions')

    def call(self, x):
        xg, xl1, xl2, xd = tf.split(x, [784, 100, 162, 8*nmax_diag], axis=-1)
        xg = tf.reshape(xg, [16, 28, 28, 1])
        xg1 = self.layer1_1(xg)
        xg1 = self.layer1_2(xg1)
        xg1 = tf.reshape(xg1, [16, 784])
        xl1 = tf.nn.relu(self.layer2_1(xl1))
        xl2 = tf.nn.relu(self.layer2_2(xl2))
        x = tf.concat((xg1, xl1, xl2), -1)
        x = self.layer3(x)
        x = self.layer4(x)
        print(x.shape)
        return x


class MNIST_CNN_PLLay(tf.keras.Model):
    def __init__(self, name='mnistcnnpllay', filters=32, kernel_size=3, unitsDense=64, unitsTopInput=32, unitsTopMiddle=64, **kwargs):
        super(MNIST_CNN_PLLay, self).__init__(name=name, **kwargs)
        self.layer1_1 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation='relu')
        self.layer1_2 = tf.keras.layers.Conv2D(1, kernel_size, padding="same", activation='relu')
        self.layer1_3 = TopoFunLayer(unitsTopMiddle, grid_size=[28, 28], tseq=np.linspace(0.05, 0.95, 18), KK=list(range(3)))
        self.layer2_1 = GThetaLayer(unitsTopInput)
        self.layer2_2 = GThetaLayer(unitsTopInput)
        self.layer3 = tf.keras.layers.Dense(unitsDense, activation='relu', name='dense_2') 
        self.layer4 = tf.keras.layers.Dense(10, name='predictions')

    def call(self, x):
        xg, xl1, xl2, xd = tf.split(x, [784, 100, 162, 8*nmax_diag], axis=-1)
        xg = tf.reshape(xg, [16, 28, 28, 1])
        xg1 = self.layer1_1(xg)
        xg1 = self.layer1_2(xg1)
        xg1 = tf.reshape(xg1, [16, 784])
        xg1_1 = tf.nn.relu(self.layer1_3(xg1))
        xg1 = tf.concat((xg1, xg1_1), -1)
        xl1 = tf.nn.relu(self.layer2_1(xl1))
        xl2 = tf.nn.relu(self.layer2_2(xl2))
        x = tf.concat((xg1, xl1, xl2), -1)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def preprocess() :
    """
    Generating necessary weight files 
    needed later to build the pretrained 
    models.
    """

    file_cn_list = [None] * nCn
    for iCn in range(nCn):
        file_cn_list[iCn] = str(int(corrupt_prob_list[iCn] * 100)).zfill(2) + \
            '_' + str(int(noise_prob_list[iCn] * 100)).zfill(2)

    x_processed_file_list = [None] * nCn
    for iCn in range(nCn):
        x_processed_file_list[iCn] = (
            'mnist_x_processed_' + file_cn_list[iCn] + '.npy')
    y_file = 'mnist_y.npy'

    model_cnn_file_array = [None] * nCn
    model_cnn_pllay_file_array = [None] * nCn
    model_cnn_pllay_input_file_array = [None] * nCn

    for iCn in range(nCn):
        model_cnn_file_array[iCn] = [None] * nTimes
        model_cnn_pllay_file_array[iCn] = [None] * nTimes
        model_cnn_pllay_input_file_array[iCn] = [None] * nTimes

    for iCn in range(nCn):
        for iTime in range(nTimes):
            file_time = str(iTime).zfill(2)
            model_cnn_file_array[iCn][iTime] = 'mnist_models/cnn_' + \
                file_cn_list[iCn] + '_' + file_time + '/model'
            model_cnn_pllay_file_array[iCn][iTime] = 'mnist_models/cnn_pllay_' + \
                file_cn_list[iCn] + '_' + file_time + '/model'
            model_cnn_pllay_input_file_array[iCn][iTime] = \
                'mnist_models/cnn_pllay_input_' + file_cn_list[iCn] + '_' + \
                file_time + '/model'

    return x_processed_file_list, y_file, model_cnn_file_array, model_cnn_pllay_file_array, model_cnn_pllay_input_file_array


def experiment(nTimes, corrupt_prob_list, noise_prob_list,
      x_processed_file_list, y_file, model_cnn_file_array,
      model_cnn_pllay_file_array, model_cnn_pllay_input_file_array,
      batch_size=16):

    print("nTimes = ", nTimes)
    (y_train, y_test) = np.load(y_file, allow_pickle=True)

    for iCn in range(nCn):
        start_time = time.time() 
        (x_train_processed, x_test_processed) = np.load(
              x_processed_file_list[iCn], allow_pickle=True)
        test_dataset = to_tf_dataset(x=x_test_processed, y=y_test,
              batch_size=batch_size)

        for iTime in range(nTimes):
  
            # CNN
            start_time_inside = time.time()
            print("CNN")
            model_cnn = MNIST_CNN()
            model_cnn.compile(optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
            model_cnn.load_weights(
                  model_cnn_file_array[iCn][iTime])
            
            output = model_cnn.predict(test_dataset)
            print("Dimension of Feature Space taken from the second layer:", output.shape)

            """
            singular_values_list = []
            
            for i in range(output.shape[0]) :

              img_list = []
              for j in range(output.shape[3]) :
                s = tf.linalg.svd(output[i,:,:,j], compute_uv=False)
                img_list.append(s)
            
              singular_values_list.append(img_list)

            singular_values_list = np.asarray(singular_values_list, dtype=np.float32)
            print("Singular Value List Shape: ", singular_values_list.shape)
            """

            #Reshaping the list. Only an array of dimension 2 can be passed through mapper algorithm due to scaling reasons
            nsamples, nx, ny, n1 = output.shape
            #singular_values_list = singular_values_list.reshape((nsamples,nx*ny))
            output_list = output.reshape((nsamples, nx*ny))

            mapper = km.KeplerMapper(verbose=2)
            projected_data = mapper.fit_transform(output_list, projection=sklearn.manifold.TSNE())

            graph = mapper.map(
                projected_data,
                output_list,
                clusterer=sklearn.cluster.KMeans()
            )            

            html = mapper.visualize(graph, path_html="kepler-mapper-output.html")
                       
            print("--- %s seconds ---" % (time.time() - start_time_inside))


if __name__ == '__main__' :

    x_processed_file_list, y_file, model_cnn_file_array, model_cnn_pllay_file_array, model_cnn_pllay_input_file_array = preprocess()

    experiment(nTimes=nTimes, corrupt_prob_list=corrupt_prob_list,
        noise_prob_list=noise_prob_list,
        x_processed_file_list=x_processed_file_list, y_file=y_file,
        model_cnn_file_array=model_cnn_file_array,
        model_cnn_pllay_file_array=model_cnn_pllay_file_array,
        model_cnn_pllay_input_file_array=model_cnn_pllay_input_file_array,
        batch_size=batch_size)