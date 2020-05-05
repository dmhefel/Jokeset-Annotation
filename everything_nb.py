import sys
import numpy as np
import os
from os.path import basename
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def evaluate(category, y_pred, y_test):
    # If category is never guessed, NumPy will produce the error "RuntimeWarning: invalid value encountered in double scalars", but the method will still run correctly
    confusion_matrix = np.zeros((2, 2))
    precision = np.zeros(2)
    recall = np.zeros(2)
    f1 = np.zeros(2)

    cat_pred = 0
    not_pred = 0
    cat_true = 0
    not_true = 0

    # get counts for metrics
    for x in range(0, len(y_test)):
        if y_test[x] == category:
            cat_true = cat_true + 1
        else:
            not_true = not_true + 1

        if y_pred[x] == category:
            cat_pred = cat_pred + 1
        else:
            not_pred = not_pred + 1

        if y_pred[x] == y_test[x] == category:
            confusion_matrix[1][1] = confusion_matrix[1][1] + 1
        elif y_test[x] == category:
            confusion_matrix[0][1] = confusion_matrix[0][1] + 1
        elif y_pred[x] == category:
            confusion_matrix[1][0] = confusion_matrix[1][0] + 1
        else:
            confusion_matrix[0][0] = confusion_matrix[0][0] + 1

    # commented out because they're not strictly needed, but are interesting if you want more data
    """
    print('Confusion matrix:\n', confusion_matrix)
    print()
    print('Pos_true', cat_true)
    print('Neg_true', not_true)
    print('Pos_pred', cat_pred)
    print('Neg_pred', not_pred)
    print()
    """

    # calculate metrics
    precision[0] = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    precision[1] = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])

    recall[0] = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    recall[1] = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])

    f1[0] = (2 * precision[0] * recall[0]) / (precision[0] + recall[0])
    f1[1] = (2 * precision[1] * recall[1]) / (precision[1] + recall[1])

    accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / (confusion_matrix[0][0] + confusion_matrix[1][0] + confusion_matrix[1][1] + confusion_matrix[0][1])

    # print results
    print('  ', category.upper())
    print('\tPrecision of', category.lower(), ':',  '{:.2f}'.format(precision[1]))
    print('\tRecall of', category.lower(), ':',  '{:.2f}'.format(recall[1]))
    print('\tF1 of', category.lower(), ':',  '{:.2f}'.format(f1[1]))
    print('\tAccuracy for', category, ':',  '{:.2f}'.format(accuracy))


def final_test(model, vectorizer):
    # perform the same steps as testing in main, but using a withheld set of 10% of our data and the model and vectorizer method that were found to produce the best results
    jokes_final = []
    categories_final = []

    for root, dirs, files in os.walk('gold_standards/final_test'):
        for name in files:
            with open(os.path.join(root, name)) as f:
                contents = f.read().lower()
                jokes_final.append(contents)

                categories_final.append(basename(root))

    features_final = vectorizer.transform(jokes_final)

    data_features_final = features_final.toarray()
    data_labels_final = categories_final

    final_pred = model.predict(data_features_final)

    print()
    print()
    print()
    print()
    print("FINAL ACCURACY:", '%.2f' % (model.score(data_features_final, data_labels_final)))
    evaluate('wordplay', final_pred, data_labels_final)
    evaluate('focus', final_pred, data_labels_final)
    evaluate('character', final_pred, data_labels_final)
    evaluate('reference', final_pred, data_labels_final)
    evaluate('shock', final_pred, data_labels_final)


if __name__ == '__main__':
    jokes = []
    categories = []

    for root, dirs, files in os.walk('gold_standards/categorized'):
        for name in files:
            with open(os.path.join(root, name)) as f:
                contents = f.read().lower()
                jokes.append(contents)

                # list of joke categories in the same order as the jokes
                categories.append(basename(root))

    # get frequency vector with x-axis jokes and y-axis words in any joke
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(jokes)

    # change features from sparse array to full array
    data_features = features.toarray()
    data_labels = categories

    x_train, x_test, y_train, y_test = train_test_split(data_features, data_labels, test_size=0.3)

    # make models and fit to training data
    model_complement = ComplementNB()
    model_complement.fit(x_train, y_train)

    model_bernoulli = BernoulliNB()
    model_bernoulli.fit(x_train, y_train)

    model_gaussian = GaussianNB()
    model_gaussian.fit(x_train, y_train)

    model_multinomial = MultinomialNB()
    model_multinomial.fit(x_train, y_train)

    comp_pred = model_complement.predict(x_test)
    bern_pred = model_bernoulli.predict(x_test)
    gaus_pred = model_gaussian.predict(x_test)
    mult_pred = model_multinomial.predict(x_test)

    # print evaluations
    print()
    print()
    print()
    print()
    print()
    print("Complement Accuracy:", '%.2f' % (model_complement.score(x_test, y_test)))
    evaluate('wordplay', comp_pred, y_test)
    evaluate('focus', comp_pred, y_test)
    evaluate('character', comp_pred, y_test)
    evaluate('reference', comp_pred, y_test)
    evaluate('shock', comp_pred, y_test)
    print()
    print()
    
    print("Bernoulli Accuracy:", '%.2f' % (model_bernoulli.score(x_test, y_test)))
    evaluate('wordplay', bern_pred, y_test)
    evaluate('focus', bern_pred, y_test)
    evaluate('character', bern_pred, y_test)
    evaluate('reference', bern_pred, y_test)
    evaluate('shock', bern_pred, y_test)
    print()
    print()

    print("Gaussian Accuracy:", '%.2f' % (model_gaussian.score(x_test, y_test)))
    evaluate('wordplay', gaus_pred, y_test)
    evaluate('focus', gaus_pred, y_test)
    evaluate('character', gaus_pred, y_test)
    evaluate('reference', gaus_pred, y_test)
    evaluate('shock', gaus_pred, y_test)
    print()
    print()

    print("Multinomial Accuracy:", '%.2f' % (model_multinomial.score(x_test, y_test)))
    evaluate('wordplay', mult_pred, y_test)
    evaluate('focus', mult_pred, y_test)
    evaluate('character', mult_pred, y_test)
    evaluate('reference', mult_pred, y_test)
    evaluate('shock', mult_pred, y_test)
    print()
    print()

    final_test(model_complement, vectorizer)
