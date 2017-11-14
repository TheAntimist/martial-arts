import os
from sklearn.externals import joblib
from sklearn.cluster import MiniBatchKMeans
import numpy as np

class MartialArts():

    labels = ["Judo", "Taekwondo"]

    path_to_videos = '/home/emanon/Desktop/Ac_BL/data/'
    DESCR_FILE = "/home/emanon/Desktop/Ac_BL/code/action-detection/Processed/descr_martial.pkl"
    FILE = "/home/emanon/Desktop/Ac_BL/code/action-detection/Processed/MA.pkl"

    def __init__(self, n_clusters=1500, verbose=False, calc_features=False, read_from_file=False):
        self.n_clusters = n_clusters
        self.verbose = verbose

        self.X = self.y = None

        if calc_features:
            from ParallelSTIP import start_stip_in_parallel
            from queue import Queue

            path_to_videos = '/home/emanon/Desktop/Ac_BL/data/Judo/Test/'  # '/home/emanon/Desktop/Ac_BL/data/Judo/Test/'
            path_to_features = path_to_videos + "descr/"

            # populate files
            if not os.path.exists(path_to_features): os.mkdir(path_to_features)

            q = Queue()
            for label in self.labels:
                path = os.path.join(self.path_to_videos, label)
                path_to_features = os.path.join(path, "descr/")
                for file in os.listdir(path):
                    if os.path.isfile(os.path.join(path, file)):
                        filename, ext = os.path.splitext(file)
                        q.put_nowait((path, filename, os.path.join(path_to_features, filename + ".txt"), ext))

            start_stip_in_parallel(q)

        if read_from_file:
            self.read_from_file()
        else:
            self.kmeans = None
            self.get_train_vocabulary()
            self.write_to_file()

    def log(self, message):
        if self.verbose:
            print(message)

    def get_stip_features_from_file(self, file):
        '''
        Assumes the file contains the features of only one single video.

        Format of features in the file:
        point-type y-norm x-norm t-norm y x t sigma2 tau2 dscr-hog(72) dscr-hof(90)

        Note, the point-type is skipped.
        :param file: File containing the STIP features.
        :return:
            position: Position of the Spatio Temporal Point on the the frames
            descriptors: HOG/HOF descriptors
        '''
        position = np.genfromtxt(file, comments='#', usecols=(4, 5, 6, 7, 8), dtype=np.int32)
        descriptors = np.genfromtxt(file, comments='#', dtype=np.float32)

        x, *y = descriptors.shape

        if not y:
            descriptors = np.reshape(descriptors, (1, x))

        descriptors = descriptors[:, 9:]

        return position, descriptors


    def get_train_vocabulary(self, n_clusters=1500):
        """
        Clusters all the spatio-temporal points and produces a vocabulary of words.
        This is then used to get a Bag of Words, for each sequence.

        :param n_clusters: number of clusters for bag of features
        :return:
        """
        DESCRIPTOR_BINS = 72 + 90
        self.log("Creating a Vocabulary for the dataset.")
        if os.path.exists(self.DESCR_FILE):
            descriptors = joblib.load(self.DESCR_FILE)
        else:
            descriptors = np.empty((0, DESCRIPTOR_BINS), dtype=np.int32)
            for label in self.labels:
                path_to_features = os.path.join(self.path_to_videos, label + "/Test/descr")
                for file in os.listdir(path_to_features):
                    if os.path.isfile(os.path.join(path_to_features, file)):
                        position, descr = self.get_stip_features_from_file(os.path.join(path_to_features, file))
                        descriptors = np.concatenate((descriptors, descr), axis=0)

            joblib.dump(descriptors, self.DESCR_FILE)

        self.log("Running KMeans on the points.")
        # KMeans to cluster the descriptors
        n_iterations = 100
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, verbose=self.verbose,
                                      max_iter=n_iterations, batch_size=1000, n_init=8)
        self.kmeans.fit(descriptors)

    def get_BOW_from_descriptors(self, descriptors, num_centroids=1500):
        '''

        :param descriptors: n x 160 matrix representing the descriptors for all points in all the videos.
        :return: Histogram of Bag of Features trained using a K-means clustering,
                  this can then be used as a Feature vector for each video
        '''

        assert self.kmeans is not None, "Vocabulary hasn't been trained over the points"

        result_cluster = self.kmeans.predict(descriptors)

        histogram = np.zeros(num_centroids, dtype=np.int32)
        for cluster in result_cluster:
            cluster = int(cluster)
            histogram[cluster] += 1
        return histogram

    def get_BOW_from_file(self, label):
        '''
        Retrieves the BOW for all the

        :param n: Number of videos to process, predefined to init the numpy matrices
        :return: X, the feature vector along with y, the array of labels
        '''

        path_to_features = os.path.join(self.path_to_videos, label + "/Test/descr")

        list_files = os.listdir(path_to_features)
        y = np.empty((len(list_files),), dtype=np.int16)
        X = np.empty((len(list_files), self.n_clusters), dtype=np.int32)
        i = 0

        for file in list_files:
            file_path = os.path.join(path_to_features, file)
            if os.path.isfile(file_path):
                self.log("Calculating Descriptor vector for file {}".format(file))
                position, descr = self.get_stip_features_from_file(file_path)
                X[i] = self.get_BOW_from_descriptors(descr, self.n_clusters)
                y[i] = self.labels.index(label)
                i += 1

        return X, y

    def dataset(self):
        """
        Returns the dataset, as set of matrices.
        :return: X, y
        """
        if self.X is not None and self.y is not None:
            return self.X, self.y

        self.X, self.y = self.get_BOW_from_file(self.labels[0])
        for label in self.labels[1:]:
            X_temp, y_temp = self.get_BOW_from_file(label)
            self.X = np.concatenate((self.X, X_temp))
            self.y = np.concatenate((self.y, y_temp))

        return self.X, self.y

    def write_to_file(self):
        X, y = self.dataset()
        joblib.dump((X, y, self.kmeans), self.FILE)

    def read_from_file(self):
        self.X, self.y, self.kmeans = joblib.load(self.FILE)
