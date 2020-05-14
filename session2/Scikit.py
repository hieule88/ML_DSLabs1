import numpy as np 
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy.sparse import csr_matrix

#-------------
"""
    load datapoints to data and labels
"""
def load_data(data_path):

    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidf = sparse_r_d.split()
        for index_tfidf in indices_tfidf:
            id = int(index_tfidf.split(':')[0])
            tfidf = float(index_tfidf.split(':')[1])
            r_d[id] = tfidf
        return r_d
    with open(data_path, 'r') as f:
        d_lines = f.read().splitlines()
    with open('../dataset/20news-bydate/words_idfs.txt', 'r') as f:
        vocab_size = len(f.read().splitlines())
    labels = []
    data = []
    
    for d in d_lines:
        label, id_doc, feature = d.split('<fff>')
        labels.append(int(label))
        r_d = sparse_to_dense(feature[2], vocab_size)
        data.append(r_d)
    return data, labels

#--------------------

def clustering_with_KMeans():
    data, labels = load_data(data_path= '../dataset/20news-bydate/data_tf_idf.txt')
    # use csr_matrix to create a sparse matrix with efficient row slicing
    X = csr_matrix(data)
    print('============')
    kmeans = KMeans(
        n_clusters= 20,
        init= 'random',
        n_init= 5, # number of time that kmeans runs with differently initialized centroids
        tol = 1e-3, # threshold for acceptable minimum error decrease
        random_state= 2018 # set to get deterministic results   
    ).fit(X)
    labels = kmeans.labels_
    return labels

def compute_accuracy(predicted_y, expected_y):
    matches = np.equal(predicted_y, expected_y)
    accuracy = np.sum(matches.astype(float)) / expected_y.size
    return accuracy

def classifying_with_linear_SVMs():
    train_X, train_y = load_data(data_path= '../dataset/20news-bydate/20news-train-tfidf.txt')
    classifier = LinearSVC(
        C= 10.0, # penalty coeff
        tol= 0.001, # tolerance for stopping criteria
        verbose= True # whether prints out logs or not
    )
    classifier.fit(train_X, train_y)

    test_X, test_y = load_data(data_path= '../dataset/20news-bydate/20news-test-tfidf.txt')
    predicted_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y = predicted_y, expected_y = test_y)
    print('Accuracy: ', accuracy)


if __name__ == "__main__":
    clustering_with_KMeans()
    classifying_with_linear_SVMs()


    
