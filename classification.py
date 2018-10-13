#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation
import assignment2



for classifier in ['myclassifier1', 'myclassifier2', 'myclassifier3']: # You may rename the names of the classifiers to something more descriptive
    if classifier == 'myclassifier1':



        print('Training ' + classifier)
        # TODO: extract features for training classifier1
        # TODO: train sentiment classifier1
    elif classifier == 'myclassifier2':

        print('Training ' + classifier)
        # TODO: extract features for training classifier2
        # TODO: train sentiment classifier2
    elif classifier == 'myclassifier3':
        print('Training ' + classifier)
        # TODO: extract features for training classifier3
        # TODO: train sentiment classifier3

    for testset in testsets.testsets:
        # TODO: classify tweets in test set
        if classifier=="myclassifier1":
            predictions = assignment2.classifier_word2vec('training.txt', testset, 80)
        if classifier=="myclassifier2":
            predictions = assignment2.classifier_ngram('training.txt', testset, 1000,800)

        if classifier=="myclassifier3":
            predictions = assignment2.classifier_doc2vec('training.txt', testset, 50)






        # TODO: Remove this line, 'predictions' should be populated with the outputs of your classifier
        evaluation.evaluate(predictions, testset, classifier)

        evaluation.confusion(predictions, testset, classifier)
