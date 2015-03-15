import numpy as np

class Preprocessor(object):

    def __init__(self):
        pass

    def build_vocabulary_and_categories(self, train_file):
        """Build the vocabulary and categories from $train_file
        """
        fin = open(train_file, 'r')
        fout1 = open('categories.txt', 'w')
        fout2 = open('vocabulary.txt', 'w')

        vocabulary = {} # key = word, value = word_id
        categories = {} # key = category, value = category_id
        word_cnt = 0
        category_cnt = 0

        for line in fin:
            terms = line.split()
            c = terms[0]
            if c not in categories:
                categories[c] = category_cnt
                category_cnt += 1
                fout1.write(c + '\n')
            for w in terms[1:]:
                if w not in vocabulary:
                    vocabulary[w] = word_cnt
                    word_cnt += 1
                    fout2.write(w + '\n')

        print 'category-size: %d' % category_cnt
        print 'vocabulary-size: %d' % word_cnt

        fin.close()
        fout1.close()
        fout2.close()

