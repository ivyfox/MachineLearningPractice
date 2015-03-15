import numpy as np

class FeatureSelector(object):
    """Feature Selector"""

    def __init__(self, train_file, categories_file = 'categories.txt', vocabulary_file = 'vocabulary.txt', ck = 500):
        self.train_file = train_file
        self.ck = ck

        self.vocabulary = {} # word --> word_id
        self.categories = {} # cate --> cate_id
        self.vocabulary_list = [] # word_id --> word
        self.categories_list = [] # cate_id --> cate

        fp1 = open(categories_file, 'r')
        cate_cnt = 0
        for line in fp1:
            cate = line.rstrip('\r\n')
            self.categories[cate] = cate_cnt
            self.categories_list.append(cate)
            cate_cnt += 1
        fp1.close()

        fp2 = open(vocabulary_file, 'r')
        word_cnt = 0
        for line in fp2:
            word = line.rstrip('\r\n')
            self.vocabulary[word] = word_cnt
            self.vocabulary_list.append(word)
            word_cnt += 1
        fp2.close()

    def select_features(self, method = 'chi2'):
        if self.ck <= 0 or self.ck > len(self.vocabulary):
            features = self.vocabulary
        elif method == 'chi2':
            features = self.chi2()
        else:
            return self.vocabulary

        fp = open('features.txt', 'w')
        for w in features: fp.write(w + '\n')
        fp.close()

    def chi2(self):
        N = 0 # num of documents
        N_c = np.zeros(len(self.categories))
        N_w = np.zeros(len(self.vocabulary))
        N_cw = np.zeros((len(self.categories), len(self.vocabulary)))

        fp = open(self.train_file)
        for line in fp:
            N += 1
            print 'processing document %d ...' % N
            terms = line.split()
            c = self.categories[terms[0]]
            N_c[c] += 1
            for t in set(terms[1:]):
                w = self.vocabulary[t]
                N_w[w] += 1
                N_cw[c][w] += 1
        fp.close()

        print 'calculating chi2 ...'
        chi2_cw = np.zeros((len(self.categories), len(self.vocabulary)))
        for c in range(len(self.categories)):
            for w in range(len(self.vocabulary)):
                N_00 = N - N_cw[c][w]
                N_01 = N_w[w] - N_cw[c][w]
                N_10 = N_c[c] - N_cw[c][w]
                N_11 = N_cw[c][w]

                E_00 = (N - N_c[c]) * (N - N_w[w]) / N
                E_01 = (N - N_c[c]) * N_w[w] / N
                E_10 = N_c[c] * (N - N_w[w]) / N
                E_11 = N_c[c] * N_w[w] / N

                chi2_cw[c][w] += (N_00 - E_00) ** 2 / E_00
                chi2_cw[c][w] += (N_01 - E_01) ** 2 / E_01
                chi2_cw[c][w] += (N_10 - E_10) ** 2 / E_10
                chi2_cw[c][w] += (N_11 - E_11) ** 2 / E_11

                print 'chi2 (%s, %s) = %.5f' % (self.categories_list[c], self.vocabulary_list[w], chi2_cw[c][w])

        feat_set =  set()
        for c in range(len(self.categories)):
            fc = chi2_cw[c].argpartition(self.ck)[-self.ck:]
            print self.categories_list[c], [self.vocabulary_list[w] for w in fc], '\n'
            feat_set |= set(fc)

        features = []
        for f_id in feat_set: features.append(self.vocabulary_list[f_id])
        return features

