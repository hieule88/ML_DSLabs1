import numpy as np
import random
from collections import defaultdict

class Member:
    """
        class of a point data (d)
    """
    def __init__(self, r_d, label=None, doc_id=None):
        self._r_d = r_d # tf-idf 
        self._label = label # cluster that d really belongs to 
        self._doc_id = doc_id # id of file 

class Cluster:
    """
        class of a label  
    """
    def __init__(self):
        self._centroid = None # type is int. 
        self._members = [] # type is Member
    
    def reset_members(self):
        self._members = [] # clear all members

    def add_member(self, member):
        self._members.append(member) 

class Kmeans:
    def __init__(self, num_clusters): 
        self._num_clusters = num_clusters
        self._clusters = [Cluster() for _ in range(self._num_clusters)] 
        self._E = [] # list of centroids
        self._S = 0 # overall similarity 

    #-------------
    """
        load data to Member
    """
    def load_data(self, data_path):
        def sparse_to_dense(sparse_r_d, vocab_size):  
            r_d = [0.0 for _ in range(vocab_size)]
            indices_tfidfs = sparse_r_d.split() 
            for index_tfidf in indices_tfidfs: 
                index = int(index_tfidf.split(':')[0])  
                tfidf = float(index_tfidf.split(':')[1])
                r_d[index] = tfidf
            return np.array(r_d)

        with open(data_path) as f:
            d_lines = f.read().splitlines() 
        with open('C:/Users/7590 inspiron/Documents/GitHub/ML_DSLabs1/dataset/20news-bydate/words_idfs.txt') as f: 
            vocab_size = len(f.read().splitlines()) 

        self._data = []
        self._label_count = defaultdict(int)
        for d in d_lines :
            features = d.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            self._label_count[label] += 1 # number of d having label "label"
            r_d = sparse_to_dense(sparse_r_d = features[2], vocab_size = vocab_size)

            self._data.append(Member(r_d = r_d, label= label, doc_id= doc_id))
    
    #---------------

    #---------------
    """
        choose centroids randomly
    """
    def random_init(self, seed_value) :
        random.seed(seed_value) # fix the centroids in the "seed_value"-th attemp
        lst_rcen = random.sample(self._data, self._num_clusters) 
        for index, cen in enumerate(lst_rcen):
            centroid = cen._r_d 
            self._clusters[index]._centroid = centroid
            self._E.append(centroid)
    
    #----------------
         
    def compute_similarity(self, member, centroid):
        dist = np.linalg.norm(member._r_d - centroid) # compute euclid distance of member and centroid
        return 1.0 / (1.0 + np.exp(-1 * dist))

    #----------------
    """
        attach a label to member
    """
    def select_cluster_for(self, member):
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self._clusters:
            similarity = self.compute_similarity(member, cluster._centroid)
            if similarity > max_similarity:
                best_fit_cluster = cluster 
                max_similarity = similarity
        best_fit_cluster.add_member(member)
        return max_similarity
    
    #-----------------

    def update_centroid_of(self, cluster):
        member_r_ds = [member._r_d for member in cluster._members]
        aver_r_d = np.mean(member_r_ds, axis =0)
        sqrt_sum_sqr = np.sqrt(np.sum(aver_r_d ** 2))
        new_centroid = np.array([value / sqrt_sum_sqr for value in aver_r_d ])

        cluster._centroid = new_centroid

    def stopping_condition(self, criterion, threshold):
        criteria = ['centroid', 'similarity', 'max_iters']
        assert criterion in criteria
        if criterion == 'max_iters':
            if self._iteration >= threshold:
                return True
            else:
                return False
        elif criterion == 'centroid':
            E_new = [list(cluster._centroid) for cluster in self._clusters]
            E_new_minus_E = [centroid for centroid in E_new if centroid not in self._E]
            self._E = E_new
            if len(E_new_minus_E) <= threshold:
                return True
            else:
                return False  
        else:
            new_S_minus_S = self._new_S - self._S
            self._S = self._new_S
            if new_S_minus_S <= threshold:
                return True
            else:
                return False
             
    def run(self, seed_value, criterion, threshold):
        self.random_init(seed_value)

        # continually update clusters until convergence
        self._iteration = 0
        while True:
            # reset clusters, retain only centroids
            for cluster in self._clusters:
                cluster.reset_members()
            self._new_S = 0
            for member in self._data:
                max_S = self.select_cluster_for(member)
                self._new_S += max_S
            for cluster in self._clusters:
                self.update_centroid_of(cluster)

            self._iteration += 1
            if self.stopping_condition(criterion, threshold):
                break
    def compute_purity(self):
        majority_sum = 0
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            max_count = max([member_labels.count(label) for label in range(20)])
            majority_sum += max_count
        return majority_sum * 1. / len(self._data)

    def compute_NMI(self):
        I_value, H_omega, H_C, N = 0., 0., 0., len(self._data)
        for cluster in self._clusters:
            wk = len(cluster._members) * 1.
            H_omega += - wk / N * np.log10(wk / N)
            member_labels = [member._label for member in cluster._members]
            for label in range(20):
                wk_cj = member_labels.count(label) * 1.
                cj = self._label_count[label]
                I_value += wk_cj / N * \
                            np.log10(N * wk_cj / (wk * cj) + 1e-12)
        for label in range(20):
            cj = self._label_count[label] * 1. 
            H_C += - cj / N * np.log10(cj / N)
        return I_value * 2. / (H_omega + H_C)


if __name__ == '__main__':
    kmeans = Kmeans(20)
    kmeans.load_data("../dataset/20news-bydate/data_tf_idf.txt")
    max_purity = 0
    best_seed_value = 0
    for seed_value in range(5):
        kmeans.run(seed_value, "max_iters", 20)
        purity = kmeans.compute_purity()
        if purity > max_purity:
            max_purity = purity
            best_seed_value = seed_value

    print(kmeans.compute_NMI())
    print(kmeans.compute_purity())