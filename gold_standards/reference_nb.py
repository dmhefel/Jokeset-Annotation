# Victoria Steger
# CS114 Spring 2020 Programming Assignment 2
# Naive Bayes Classifier and Evaluation

import os
import numpy as np
from collections import defaultdict


class NaiveBayes():

    def __init__(self):
        # be sure to use the right class_dict for each data set
        self.class_dict = {0: 'neg', 1: 'pos'}
        # self.class_dict = {0: 'action', 1: 'comedy'}
        self.feature_dict = {}
        self.prior = None
        self.likelihood = None
        # value for add k smoothing
        self.k = 3

    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    Specifically, fills in self.prior and self.likelihood such that:
    self.prior[class] = log(P(class))
    self.likelihood[class][feature] = log(P(feature|class))
    '''
    def train(self, train_set):
        # list of punctuation to use for not-prepending
        punct = ['.', ',', '!', '?', ':', ';']
        self.feature_dict = self.select_features(train_set)

        # variables for document counts
        total = 0
        neg = 0
        pos = 0

        # dictionaries for word counts in both categories
        pos_vocab = dict()
        neg_vocab = dict()

        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    total = total + 1
                    contents = f.read().lower()
                    words = contents.split()

                    # when 'not' is found, prepend not- to every word between 'not' and the next punctuation
                    i = 0
                    while (i < len(words)):
                        if(words[i] == 'not'):
                            while (((i+1) < len(words)) and (words[i+1] not in punct)):
                                i += 1
                                words[i] = "not_" + words[i]
                        i += 1

                    if (root[-3:] == 'pos'):
                        pos = pos + 1
                    else:
                        neg = neg + 1

                    # get word counts after not- prepending
                    for word in words:
                        if word not in pos_vocab:
                            pos_vocab[word] = 0

                        if word not in neg_vocab:
                            neg_vocab[word] = 0

                        if (root[-3:] == 'pos'):
                            pos_vocab[word] = pos_vocab[word] + 1

                        if (root[-3:] == 'neg'):
                            neg_vocab[word] = neg_vocab[word] + 1

        # normalize counts to probabilities, and take logs
        self.prior = ([neg/total, pos/total])
        self.prior = np.log(self.prior)

        self.likelihood = np.zeros((len(self.class_dict), len(self.feature_dict)))

        neg_words = sum(neg_vocab.values())
        pos_words = sum(pos_vocab.values())

        for i in self.class_dict:
            for j in self.feature_dict:
                # use count of feature word + 1*k if present, and 1*k otherwise for plus k smoothing
                if (self.feature_dict[j] in pos_vocab) and (self.class_dict[i] == 'pos'):
                    self.likelihood[i][j] = (pos_vocab[self.feature_dict[j]] + self.k*1) / (pos_words + self.k*len(pos_vocab))
                elif (self.feature_dict[j] not in pos_vocab) and (self.class_dict[i] == 'pos'):
                    self.likelihood[i][j] = self.k*1 / (pos_words + self.k*len(pos_vocab))
                elif (self.feature_dict[j] in neg_vocab) and (self.class_dict[i] == 'neg'):
                    self.likelihood[i][j] = (neg_vocab[self.feature_dict[j]] + self.k*1) / (neg_words + self.k*len(neg_vocab))
                elif (self.feature_dict[j] not in neg_vocab) and (self.class_dict[i] == 'neg'):
                    self.likelihood[i][j] = self.k*1 / (neg_words + self.k*len(neg_vocab))

        self.likelihood = np.log(self.likelihood)

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        punct = ['.', ',', '!', '?', ':', ';']
        results = defaultdict(dict)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    contents = f.read().lower()
                    words = contents.split()
                    feature = np.zeros(len(self.feature_dict))

                    # when 'not' is found, prepend not- to every word between 'not' and the next punctuation
                    i = 0
                    while (i < len(words)):
                        if(words[i] == 'not'):
                            while (((i+1) < len(words)) and (words[i+1] not in punct)):
                                i += 1
                                words[i] = "not_" + words[i]
                        i += 1

                    # get word counts for all words after not- prepending
                    vocab = dict()
                    for word in words:
                        if word in vocab:
                            vocab[word] = vocab[word] + 1
                        else:
                            vocab[word] = 1

                    # take feature counts from word count dictionary to avoid iterating over full review multiple times
                    for i in self.feature_dict:
                        if self.feature_dict[i] in vocab:
                            feature[i] = vocab[self.feature_dict[i]]
                        else:
                            feature[i] = 0

                    if root[-3:] == 'neg':
                        results[name]['correct'] = 0
                    else:
                        results[name]['correct'] = 1

                    temp = np.dot(self.likelihood, feature)
                    temp = temp + self.prior
                    results[name]['predicted'] = np.argmax(temp, axis=None)

        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        confusion_matrix = np.zeros((len(self.class_dict), len(self.class_dict)))
        precision = np.zeros(len(self.class_dict))
        recall = np.zeros(len(self.class_dict))
        f1 = np.zeros(len(self.class_dict))

        pos_pred = 0
        neg_pred = 0
        pos_true = 0
        neg_true = 0

        for file in results:
            if results[file]['correct'] == 0:
                neg_true = neg_true + 1

            if results[file]['correct'] == 1:
                pos_true = pos_true + 1

            if results[file]['predicted'] == 0:
                neg_pred = neg_pred + 1

            if results[file]['predicted'] == 1:
                pos_pred = pos_pred + 1

            if results[file]['predicted'] == results[file]['correct'] == 0:
                confusion_matrix[0][0] = confusion_matrix[0][0] + 1
            elif (results[file]['predicted'] == 0) and (results[file]['correct'] == 1):
                confusion_matrix[0][1] = confusion_matrix[0][1] + 1
            elif (results[file]['predicted'] == 1) and (results[file]['correct'] == 0):
                confusion_matrix[1][0] = confusion_matrix[1][0] + 1
            elif results[file]['predicted'] == results[file]['correct'] == 1:
                confusion_matrix[1][1] = confusion_matrix[1][1] + 1

        # while not required by the project, I found these pieces of information useful for understanding the results
        """
        print('Confusion matrix:\n', confusion_matrix)
        print()
        print('Pos_true', pos_true)
        print('Neg_true', neg_true)
        print('Pos_pred', pos_pred)
        print('Neg_pred', neg_pred)
        print()
        """

        precision[0] = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
        precision[1] = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])

        recall[0] = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
        recall[1] = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])

        f1[0] = (2 * precision[0] * recall[0]) / (precision[0] + recall[0])
        f1[1] = (2 * precision[1] * recall[1]) / (precision[1] + recall[1])

        accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / (confusion_matrix[0][0] + confusion_matrix[1][0] + confusion_matrix[1][1] + confusion_matrix[0][1])

        print('NEGATIVE CATEGORY')
        print('Precision of', self.class_dict[0], ': ', '{:.2f}'.format(precision[0]))
        print('Recall of', self.class_dict[0], ':',  '{:.2f}'.format(recall[0]))
        print('F1 of', self.class_dict[0], ':',  '{:.2f}'.format(f1[0]))
        print()
        print('POSITIVE CATEGORY')
        print('Precision of', self.class_dict[1], ':',  '{:.2f}'.format(precision[1]))
        print('Recall of', self.class_dict[1], ':',  '{:.2f}'.format(recall[1]))
        print('F1 of', self.class_dict[1], ':',  '{:.2f}'.format(f1[1]))
        print()
        print('Total accuracy:',  '{:.2f}'.format(accuracy))

    '''
    Performs feature selection.
    Returns a dictionary of features.
    '''
    def select_features(self, train_set):
        """
        # initial method
        return {0: 'fast', 1: 'couple', 2: 'shoot', 3: 'fly'}
        """
        """
        # the total vocabulary of both review types
        vocab = set()
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    contents = f.read().lower()
                    words = set(contents.split())
                    vocab = vocab.union(words)

        index = list(range(0, len(vocab)))
        vocab = list(vocab)
        dicto = dict(zip(index, vocab))
        # print('Vocab length is', len(vocab))
        return dicto
        """
        """
        # the total vocabulary of both review types with not prepended to all words after 'not' and before a piece of punctuation
        punct = ['.', ',', '!', '?', ':', ';']
        vocab = set()
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    contents = f.read().lower()
                    words = contents.split()

                    i = 0
                    while (i < len(words)):
                        if(words[i] == 'not'):
                            while (((i+1) < len(words)) and (words[i+1] not in punct)):
                                i += 1
                                words[i] = "not_" + words[i]
                        i += 1

                    vocab = vocab.union(set(words))

        index = list(range(0, len(vocab)))
        vocab = list(vocab)
        dicto = dict(zip(index, vocab))
        # print('Vocab length is', len(vocab))
        return dicto
        """

        # the total vocabulary of negative reviews with not prepended to all words after 'not' and before a piece of punctuation
        punct = ['.', ',', '!', '?', ':', ';']
        vocab = set()
        # change '/neg' to 'pos' below to run against only the positive review vocabulary
        train_set = train_set + '/neg'
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    contents = f.read().lower()
                    words = contents.split()

                    # when 'not' is found, prepend not- to every word between 'not' and the next punctuation
                    i = 0
                    while (i < len(words)):
                        if(words[i] == 'not'):
                            while (((i+1) < len(words)) and (words[i+1] not in punct)):
                                i += 1
                                words[i] = "not_" + words[i]
                        i += 1

                    # take vocab after not-prepending
                    vocab = vocab.union(set(words))

        index = list(range(0, len(vocab)))
        vocab = list(vocab)
        dicto = dict(zip(index, vocab))
        # while not required, I found this measure useful while testing
        # print('Vocab length is', len(vocab))
        return dicto


if __name__ == '__main__':
    nb = NaiveBayes()
    # make sure these point to the right directories
    nb.train('movie_reviews/train')
    # nb.train('movie_reviews_small/train')
    results = nb.test('movie_reviews/dev')
    # results = nb.test('movie_reviews_small/test')
    nb.evaluate(results)
