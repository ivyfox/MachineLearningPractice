import numpy as np

class Classifier(object):

    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.vocabulary = {} # key = word, value = word_id
        self.categories = {} # key = category, value = category_id

    def build_vocabulary_and_categories(self):
        """Build the vocabulary and categories from self.train_file
        """

        word_id = 0
        category_id = 0
        fp = open(self.train_file)
        for line in fp:
            terms = line.split()
            c = terms[0]
            if c not in self.categories:
                self.categories[c] = category_id
                category_id += 1
            for w in terms[1:]:
                if w not in self.vocabulary:
                    self.vocabulary[w] = word_id
                    word_id += 1
        fp.close()

        print 'vocabulary-size: %d' % len(self.vocabulary)
        print 'category-size: %d' % len(self.categories)

