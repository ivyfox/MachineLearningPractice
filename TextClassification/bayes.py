import numpy as np

from basic import Classifier

class BayesClassifier(Classifier):
    """Naive-Bayes-Classifier"""

    def __init__(self, train_file, test_file, model = 'bernoulli'):
        super(BayesClassifier, self).__init__(train_file, test_file)

        self.model = model

        print 'building vocabulary ...'
        super(BayesClassifier, self).build_vocabulary_and_categories()

        self.N_c = np.zeros(len(self.categories), dtype = int)
        self.N_cw = np.zeros((len(self.categories), len(self.vocabulary)), dtype = int)
        self.F_c = np.zeros(len(self.categories), dtype = int)
        self.F_cw = np.zeros((len(self.categories), len(self.vocabulary)), dtype = int)

        self.priori = np.zeros(len(self.categories), dtype = float)

        if model == 'bernoulli':
            self.likelihood = np.zeros((2, len(self.categories), len(self.vocabulary)), dtype = float)
        elif model == 'multinomial':
            self.likelihood = np.zeros((len(self.categories), len(self.vocabulary)), dtype = float)
        else:
            raise Exception('Unknow model name.')

    def get_features(self, words):
        """Get features form words sequence
        """

        features = {}
        for w in words: 
            if w in self.vocabulary:
                w_id = self.vocabulary[w]
                if self.model == 'bernoulli':
                    features[w_id] = 1
                elif self.model == 'multinomial':
                    if w_id in features: features[w_id] += 1
                    else: features[w_id] = 1
        return features

    def train(self):
        """Learn from the train_set
            
            Every line in self.train_file should be in this form:
              label w1 w2 ...
        """
        print 'training ...'

        if self.model == 'bernoulli':

            fp = open(self.train_file)
            for line in fp:
                terms = line.split()
                c = self.categories[terms[0]]
                features = self.get_features(terms[1:])
                self.N_c[c] += 1
                for w in features: self.N_cw[c][w] += 1
            fp.close()

            self.priori = np.log(self.N_c * 1.0 / sum(self.N_c))
            for c in range(len(self.categories)):
                self.likelihood[1][c] = np.log((self.N_cw[c] + 1.0) / (self.N_c[c] + 2.0))
                self.likelihood[0][c] = np.log(1.0 - (self.N_cw[c] + 1.0) / (self.N_c[c] + 2.0))

        elif self.model == 'multinomial':

            fp = open(self.train_file)
            for line in fp:
                terms = line.split()
                c = self.categories[terms[0]]
                features = self.get_features(terms[1:])
                self.N_c[c] += 1
                self.F_c[c] += sum(features.values())
                for w in features: self.F_cw[c][w] += features[w]
            fp.close()

            self.priori = np.log(self.N_c * 1.0 / sum(self.N_c))
            for c in range(len(self.categories)):
                self.likelihood[c] = np.log((self.F_cw[c] + 1.0) / (self.F_c[c] + len(self.vocabulary)))

        else:
            pass

    def classify(self, words):
        posterior = {}
        for c in range(len(self.categories)):
            posterior[c] = self.priori[c]
            features = self.get_features(words)
            if self.model == 'bernoulli':
                for w in range(len(self.vocabulary)):
                    if w in features: posterior[c] += self.likelihood[1][c][w]
                    else: posterior[c] += self.likelihood[0][c][w]
            elif self.model == 'multinomial':
                for w in features:
                    posterior[c] += features[w] * self.likelihood[c][w]
            else:
                pass
        return max(posterior, key = posterior.get)

    def test(self):
        cnt_all = 0
        cnt_correct = 0
        fp = open(self.test_file)
        for line in fp:
            terms = line.split()
            c1 = self.categories[terms[0]]
            c2 = self.classify(terms[1:])
            cnt_all += 1
            if c1 == c2: cnt_correct += 1
            print 'test %d: %d, %d (%.5f)' % (cnt_all, c1, c2, cnt_correct * 100.0 / cnt_all)
        fp.close()
        print 'accuracy: %.5f%%' % (cnt_correct * 100.0 / cnt_all)
    
    def run(self):
        self.train()
        self.test()

