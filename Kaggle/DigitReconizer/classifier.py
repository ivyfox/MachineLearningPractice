import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import cross_validation, metrics, decomposition
from sklearn import neighbors, linear_model, multiclass, svm

IMAGE_SIZE = 28


def load_train(train_filepath):
    df = pd.read_csv(train_filepath)
    mtrx = df.as_matrix()
    X = np.where(mtrx[:, 1:] >= 128, 1, 0)
    y = mtrx[:, 0]
    return X, y


def draw_digit(x):
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    for i, p in enumerate(x):
        img[i / IMAGE_SIZE, i % IMAGE_SIZE] = p * 128
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.show()


def evaluate_model(y_test, y_pred):
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))


def generate_predict(test_filepath, predict_filepath, clf, pca):
    df = pd.read_csv(test_filepath)
    X = pca.transform(np.where(df.as_matrix() >= 128, 1, 0))
    y = clf.predict(X)

    fp = open(predict_filepath, 'w')
    fp.write('ImageId,Label\n')
    for i, t in enumerate(y):
        fp.write('%d,%d\n' % (i + 1, t))
    fp.close()


def main():
    X, y = load_train('train.csv')

    pca = decomposition.PCA(n_components=64)
    X = pca.fit_transform(X)

    clf = svm.SVC(verbose=True)

    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    #     X, y, test_size=0.3, random_state=42)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # evaluate_model(y_test, y_pred)

    clf.fit(X, y)
    generate_predict('test.csv', 'predict.csv', clf, pca)


main()
