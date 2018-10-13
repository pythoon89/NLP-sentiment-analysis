1
# coding: utf-8

# In[49]:


import string
import unicodedata
import re
import nltk
import numpy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from sklearn.feature_selection import VarianceThreshold
from nltk import *
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def preprocess(filename):
    text = []
    features = []
    label = []
    idn = []
    Stopwords = stopwords.words('english')
    # stopwords list
    for j in range(0, len(Stopwords)):
        Stopwords[j] = re.sub('[^0-9a-zA-Z\s]', "", Stopwords[j])

    # read files
    with open(filename, 'r', encoding='UTF-8') as r:
        for line in r:
            line = line[0:-1].split('\t', 2)
            text.append(line)
        for i in range(len(text)):
            text[i][2] = text[i][2].encode('utf-8').decode('unicode_escape', 'ignore').lower()
# convert letters to lowercase
            text[i][2] = re.sub(r'[a-zA-Z]+://[\S]+|[\S]+[.][\S]+[.][\S]+', '', text[i][2])
            text[i][2] = re.sub(r'[^a-z 0-9?]+', '', text[i][2])
            text[i][2] = re.sub(r'^[0-9]+[\s]|(?<![a-z0-9])[^a-z]+(?![a-z0-9])|[\s][0-9]+$', '', text[i][2])
            text[i][2] = text[i][2].lower().replace("?", " ?")

            text[i][2] = text[i][2].split()
            temp = text[i][2]
            features.append(temp)
            label.append(text[i][1])
            idn.append(text[i][0])

            # remove stop words and lemmatise
            #     lemmatizer=WordNetLemmatizer()
    final_features = []
    count = 0
    for item in features:
        final_features.append([])
        for element in item:
            #             element=lemmatizer.lemmatize(element)
            if element not in Stopwords:
                final_features[count].append(element)
        count += 1

    return final_features, label, idn


import gensim
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec


def convertfeature(list1, model):
    new1 = []
    new2 = []
    n = model.layer1_size
    for j in range(len(list1)):
        new1.append(list1[j])
        for i in range(len(list1[j])):
            try:
                new1[j][i] = model[list1[j][i]]
            except KeyError:
                #                 print (item[i],'not in vocabulary')
                new1[j][i] = [0] * n

    count = 0
    for item in new1:
        new2.append([0] * n)
        sum = 0
        for j in range(n):

            for i in item:
                #                 print(i)
                #                 print(i[j])
                sum = sum + i[j]
            new2[count][j] = sum / len(i)
        count += 1

    return new2


            



def naive_bayes_classifier(train_x, train_y):    
       
    model = MultinomialNB(alpha=0.01)    
    model.fit(train_x, train_y)    
    return model 
    
# KNN Classifier    
def knn_classifier(train_x, train_y):    
    from sklearn.neighbors import KNeighborsClassifier    
    model = KNeighborsClassifier()    
    model.fit(train_x, train_y)    
    return model


# SVM Classifier using cross validation    
def svm_cross_validation(train_x, train_y):    
#     from sklearn.grid_search import GridSearchCV    
    from sklearn.svm import SVC    
    model = SVC(kernel='rbf', probability=True)    
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}    
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)    
    grid_search.fit(train_x, train_y)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)  
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    model.fit(train_x, train_y)    
    return model
# Decision Tree Classifier    
def decision_tree_classifier(train_x, train_y):    
    from sklearn import tree    
    model = tree.DecisionTreeClassifier()    
    model.fit(train_x, train_y)    
    return model
# Logistic Regression Classifier    
def logistic_regression_classifier(train_x, train_y):    
    from sklearn.linear_model import LogisticRegression    
    model = LogisticRegression(penalty='l1')
    model.fit(train_x, train_y)    
    return model 
    
# SVM Classifier    
def svm_classifier(train_x, train_y):    
    from sklearn.svm import SVC    
    model = SVC(kernel='rbf', probability=True)    
    model.fit(train_x, train_y)    
    return model

def bigramlist(file):
    bi=[]
    for item in file:
        item=list(nltk.bigrams(item))
#         for i in range(len(item)):
#             item[i]=(item[i][0]+' '+item[i][1])

        bi.append(item)
    return bi


def bigram_vec(file, bifile, n):
    from nltk import bigrams
    list_bi = []

    list_str = []
    for item in file:
        for i in item:
            list_str.append(i)
    # use bigrams to build bigrams model

    bigramlist = list(nltk.bigrams(list_str))

    f = FreqDist(bigramlist)
    fsorted = sorted(f.items(), key=lambda x: x[1], reverse=True)[0:n]

    count = 0
    for item1 in bifile:
        list_bi.append([])

        for j in range(0, n):

            if fsorted[j][0] in item1:
                list_bi[count].append(1)
            else:
                list_bi[count].append(0)
        count += 1

    return list_bi
def bigram_sen(file):

    bi=[]
    for item in file:
        item=list(nltk.bigrams(item))
        for i in range(len(item)):
            item[i]=(item[i][0]+' '+item[i][1])

        bi.append(item)
    return bi

def doc2v_file(word):
    import gensim.models.doc2vec
    from gensim.models.doc2vec import Doc2Vec, LabeledSentence
    from gensim.models.doc2vec import TaggedDocument

    sen=[]
    for i in range(len(word)):
        T = TaggedDocument(word[i],[i])
        sen.append(T)
    return sen

def classifier_ngram(trainset,testset,n1,n2):
    from sklearn.feature_selection import VarianceThreshold
    from nltk import precision
    from sklearn import metrics


    train_f, train_l, train_id = preprocess(trainset)
    test_f,test_l,test_id=preprocess(testset)

    train_f2 = bigramlist(train_f)

    test_f2 = bigramlist(test_f)
    train_f1 = bigram_vec(train_f, train_f2, n1)
    test_f1 = bigram_vec(train_f, test_f2, n1)



    pca = PCA(n_components=n2)
    train_f1 = pca.fit_transform(train_f1)
    test_f1 = pca.transform(test_f1)
# //*******************************************************************
    import pickle
    modelfile = 'ngramLR.model'
    if os.path.isfile("ngramLR.model"):

        f = open(modelfile, 'rb')
        model_LR = pickle.load(f)

    else:
        model_LR = logistic_regression_classifier(train_f1, train_l)
        f = open(modelfile, 'wb')
        pickle.dump(model_LR, f)
        f.close()


    predict = model_LR.predict(test_f1)
    # accuracy = metrics.accuracy_score(test_l, predict)
    # print('accuracy: %.2f%%' % (100 * accuracy))

    predict = dict(zip(test_id, predict))


    return predict


def classifier_word2vec(trainset,testset,n):
    train_f,train_l,train_id=preprocess(trainset)
    test_f,test_l,test_id=preprocess(testset)

    if os.path.isfile("w2v.model"):
        model1 = Word2Vec.load('w2v.model')
    else:
        model1 = gensim.models.Word2Vec(train_f, size=n,window=6, min_count=30, workers=2, iter=20)
        model1.save('w2v.model')

    train_f=convertfeature(train_f,model1)
    test_f=convertfeature(test_f,model1)
# ****************************************************LR
    import pickle
    modelfile = 'w2vLR.model'
    if os.path.isfile("w2vLR.model"):

        f = open(modelfile, 'rb')
        model_LR = pickle.load(f)

    else:
        model_LR = logistic_regression_classifier(train_f, train_l)
        f = open(modelfile, 'wb')
        pickle.dump(model_LR, f)
        f.close()

    predict = model_LR.predict(test_f)

    predict = dict(zip(test_id, predict))
    return predict

def classifier_doc2vec(trainset,testset,n):
    import gensim.models.doc2vec
    from gensim.models.doc2vec import Doc2Vec, LabeledSentence
    from gensim.models.doc2vec import TaggedDocument
    from nltk import precision
    from sklearn import metrics
    train_f, train_l, train_id = preprocess(trainset)
    test_f, test_l, test_id = preprocess(testset)
    sen = doc2v_file(train_f)

    if os.path.isfile("d2v.model"):
        model1 = Word2Vec.load('d2v.model')
    else:
        model1 = Doc2Vec(sen, size=50, window=5, min_count=1, workers=2)
        model1.save('d2v.model')

    train_f = convertfeature(train_f, model1)
    test_f = convertfeature(test_f, model1)


    import pickle
    modelfile = 'd2vLR.model'
 # save model in file
    if os.path.isfile("d2vLR.model"):

        f = open(modelfile, 'rb')
        model_LR = pickle.load(f)

    else:
        model_LR = logistic_regression_classifier(train_f, train_l)
        f = open(modelfile, 'wb')
        pickle.dump(model_LR, f)
        f.close()

    predict = model_LR.predict(test_f)

    # accuracy = metrics.accuracy_score(test_l, predict)
    # print('accuracy: %.2f%%' % (100 * accuracy))
    predict = dict(zip(test_id, predict))
    return predict


# classifier_word2vec('training.txt','dev.txt',80)
# classifier_ngram('training.txt','dev.txt',1000,800)
# classifier_doc2vec('training.txt','dev.txt',50)













#
# model = classifiers['KNN'](train_f, train_l)
# predict = model.predict(test_f)
# # precision = metrics.precision_score(test_l, predict)
# # recall = metrics.recall_score(test_l, predict)
# # print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
# accuracy = metrics.accuracy_score(test_l, predict)
# print('accuracy: %.2f%%' % (100 * accuracy))
    


# # In[ ]:
#
#
# model = classifiers['SVM'](train_f, train_l)
# predict = model.predict(test_f)
# # precision = metrics.precision_score(test_l, predict)
# # recall = metrics.recall_score(test_l, predict)
# # print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
# accuracy = metrics.accuracy_score(test_l, predict)
# print('accuracy: %.2f%%' % (100 * accuracy))



