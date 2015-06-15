import parser
import os
import scipy
import scipy.sparse
import scipy.sparse.linalg
import math
import numpy as np
import subprocess
import porterStemmer as ps
import time

dirName = 'text'
docNames = []
translation_table = dict.fromkeys(map(ord, '!:,;\"\'\r\n()%?.`<>-'), None)
kNum = 5
k = 4

def update_progress(progress, max, time):
    proc = (100 * progress) / (max-1)
    print('\r[{0}]{4}% {1}/{2} time: {3}ms'.format('#' * int(proc / 5), progress, max - 1, time * 1000, proc), end="")

#parsowanie i zwracanie bag_of_words i term_by_document
#1, 2, 3
def parse(filename, term_by_document):
    bag_of_words = dict()
    f = subprocess.check_output(["./stemmer", filename]).decode("utf-8").splitlines()

    for i in f:
        words = i.split(' ')
        for word in range(len(words)):
            words[word] = words[word].translate(translation_table).lower()
            term_by_document.add(words[word])

        for word in words:
            if word in bag_of_words.keys():
                bag_of_words[word] += 1
            else:
                bag_of_words[word] = 1
    return bag_of_words, term_by_document

def makeTermAndBags(directory, term):
    i = 0
    bag_of_words_list = [{}] * len(os.listdir(directory))
    start_time = time.time()
    for name in os.listdir(directory):
        currFile = directory + "/" + name
        if os.path.isfile(currFile):
            docNames.append(currFile)
            bag_of_words_list[i], term = parse(currFile, term)
            i += 1
        update_progress(i, len(os.listdir(directory)), time.time() - start_time)
    return bag_of_words_list, term

#4
def create_A_matrix(bag_of_words, term_by_document):
    A = scipy.sparse.dok_matrix((len(term_by_document), len(bag_of_words)))
    mapper = {}
    i = 0
    start_time = time.time()
    for word in term_by_document:
        mapper[word] = i
        i += 1

    for j in range(len(bag_of_words)):
        for word in bag_of_words[j].keys():
            A[mapper[word], j] = bag_of_words[j][word]
            update_progress(j, len(bag_of_words), time.time() - start_time)
    del bag_of_words
    return A, mapper

#5
def idf_for_bag_list(bag_of_word_list, term_by_document):
    idf = {}
    start_time = time.time()
    for word in term_by_document:
        idf[word] = idf_for_word(word, bag_of_word_list)
    for i in range(len(bag_of_word_list)):
        for word in bag_of_word_list[i].keys():
            bag_of_word_list[i][word] *= 1.0 * idf[word]
        update_progress(i, len(bag_of_word_list), time.time() - start_time)
    return bag_of_word_list


def idf_for_word(word, bagList):
    N = len(bagList)
    nw = 0.0
    for i in range(N):
        if word in bagList[i] and bagList[i][word] > 0:
            nw += 1.0
    return math.log(N * 1.0 / nw)

#6
def probability(q, d):
    qT = np.array(q).transpose()
    try:
        matrix = np.dot(qT, d) / (np.linalg.norm(qT) * np.linalg.norm(d))
    except:
        pass

    return matrix

#7
def probability2(q, A, i):
    ei = [0] * len(A)

    for k in range(len(ei)):
        ei[k] = A[k][i]

    qT = np.array(q).transpose()
    ei = np.array(ei)
    return qT.dot(ei) / (np.linalg.norm(qT) * np.linalg.norm(ei))


def query(matrix, data, A):
    (x,y) = A.get_shape()
    for i in range(y):
        b = [0] * x
        for c in range(x):
            b[c] = A[c, i]
        data[i] = probability(matrix, b)

    return data


def querySVD(matrix, probs, A, U, S, V, svdCount):
    S2 = S
    for i in range(svdCount + 1, len(S2)):
        S2[i] = 0
    nA = U.dot(np.diag(S2)).dot(V)
    for i in range(A.get_shape()[1]):
        probs[i] = probability2(matrix, nA, i)
    del nA
    return probs


"""
Przetwarza zdanie wejsciowe na wektor bag-of-words.
"""


def preProcess2(q, A, mapper, term):
    p = dict()
    #Najpierw stemming
    q = preProcessQuery(q)

    for k in term:
        p[k] = 0
    for k in q:
        if k in p:
            p[k] += 1

    matrix = [0] * len(mapper)
    probs = [0] * A.get_shape()[1]
    for k in p:
        matrix[mapper[k]] = p[k]
    return matrix, probs


"""
Wyszukuje indeks dla maksymalnej wartosci
"""


def findMaxIndex(result):
    maxer = max(result)
    for i in range(len(result)):
        if result[i] == maxer:
            return i
    return 0


"""
Wykonuje stemming na wektorze wejsciowym
"""


def preProcessQuery(q):
    stem = ps.PorterStemmer()
    for k in range(len(q)):
        q[k] = q[k].translate(translation_table).lower()
        q[k] = stem.stem(q[k], 0, len(q[k]) - 1)
    return q


"""
Zwraca 5 najlepszych wynikow.
"""


def giveAnswer(probs):
    const = 5
    b = list(probs)
    b.sort()
    results = [0] * const
    j = 0
    for i in range(len(b) - const, len(b)):
        results[j] = b[i]
        j += 1

    resultsNames = [dict()] * const
    for i in range(len(results)):
        resultsNames[i] = {docNames[probs.index(results[i])], probs[probs.index(results[i])]}

    return resultsNames


term = set()
print("Czytanie i parsowanie")
bagList, term = makeTermAndBags(dirName, term)
print("\nPrzetwarzanie inverse document frequency")
bagList = idf_for_bag_list(bagList, term)
print("\nTworzenie macierzy A")
A, mapper = create_A_matrix(bagList, term)
print("\nSVD")
#8
start_time = time.time()
U, S, V = scipy.sparse.linalg.svds(A, k=kNum)
print("Czas: " + str(time.time() - start_time))
print("Gotowe")
q = input("Wyszukaj: ").split(' ')
matrix, probs = preProcess2(q, A, mapper, term)
result = query(list(matrix), probs, A)
print("Dokladnosc: " + str(max(result)))
print("Nazwa pliku: " + docNames[findMaxIndex(result)])
print("Inne wyniki: " + str(giveAnswer(result)))
print("Uzywajac SVD:")
result = querySVD(list(matrix), probs, A, U, S, V, k)
print("Dokladnosc: " + str(max(result)))
print("Nazwa pliku: " + docNames[findMaxIndex(result)])
print(giveAnswer(result))