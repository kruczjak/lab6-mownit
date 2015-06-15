import os
from scipy.sparse import dok_matrix, linalg
import math
import numpy as np
import subprocess
import porterStemmer as ps
import time
import warnings

warnings.filterwarnings('ignore')

dirName = 'text'
translation_table = dict.fromkeys(map(ord, '\\!{}:,;\"\'\r\n()%?.`<>-'), None)
SVDk1 = 5
SVDk2 = 4

def update_progress(progress, max, time):
    percent = (100 * progress) / (max-1)
    print('\r[{0}]{4}% {1}/{2} time: {3}ms'.format('#' * int(percent / 5), progress, max - 1, time * 1000, percent), end="")


#1, 2, 3
def parse(filename, term_by_document):
    bag_of_words = {}
    f = subprocess.check_output(["./stem", filename]).decode("utf-8").splitlines()

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

def read_and_parse(directory, term):
    i = 0
    bag_of_words_list = [{}] * len(os.listdir(directory))
    start_time = time.time()
    for name in os.listdir(directory):
        file_name = directory + "/" + name
        bag_of_words_list[i], term = parse(file_name, term)
        i += 1
        update_progress(i-1, len(os.listdir(directory)), time.time() - start_time)
    return bag_of_words_list, term

#4
def create_A_matrix(bag_of_words, term_by_document):
    A = dok_matrix((len(term_by_document), len(bag_of_words)))
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


def idf_for_word(word, bag_of_words_list):
    N = len(bag_of_words_list)
    nw = 0.0
    for i in range(N):
        if word in bag_of_words_list[i] and bag_of_words_list[i][word] > 0:
            nw += 1.0
    return math.log(N * 1.0 / nw)

#6, 7
def probability_first_eq(q, d):
    qT = np.array(q).transpose()
    return np.dot(qT, d) / (np.linalg.norm(qT) * np.linalg.norm(d))

#8
def probability_second_eq(q, A, i):
    ei = [0] * len(A)

    for k in range(len(ei)):
        ei[k] = A[k][i]

    qT = np.array(q).transpose()
    ei = np.array(ei)
    return qT.dot(ei) / (np.linalg.norm(qT) * np.linalg.norm(ei))


def query(matrix, data, A):
    (x,y) = A.get_shape()
    start_time = time.time()
    for i in range(y):
        b = [0] * x
        for c in range(x):
            b[c] = A[c, i]
        data[i] = probability_first_eq(matrix, b)
        update_progress(i, y, time.time() - start_time)
    return data


def querySVD(matrix, probs, A, U, S, V, svdCount):
    S2 = S
    for i in range(svdCount + 1, len(S2)):
        S2[i] = 0
    nA = U.dot(np.diag(S2)).dot(V)
    for i in range(A.get_shape()[1]):
        probs[i] = probability_second_eq(matrix, nA, i)
    del nA
    return probs


def process_query(q, A, mapper, term):
    p = {}
    q = parse_query(q)

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

def parse_query(q):
    stem = ps.PorterStemmer()
    for k in range(len(q)):
        q[k] = q[k].translate(translation_table).lower()
        q[k] = stem.stem(q[k], 0, len(q[k]) - 1)
    return q

def prepareOutput(data):
    how_many = 5
    j = 0
    b = list(data)
    b.sort()
    results = [0] * how_many
    for i in range(len(b) - how_many, len(b)):
        results[j] = b[i]
        j += 1

    resultsOut = [{}] * how_many
    for i in range(len(results)):
        resultsOut[i] = {data[data.index(results[i])]}

    return resultsOut

main_start_time = time.time()
term_by_document = set()
print("Czytanie i parsowanie")
bag_of_words_list, term_by_document = read_and_parse(dirName, term_by_document)
print("\nPrzetwarzanie inverse document frequency")
bag_of_words_list = idf_for_bag_list(bag_of_words_list, term_by_document)
print("\nTworzenie macierzy A")
A, mapper = create_A_matrix(bag_of_words_list, term_by_document)
print("\nSVD")
#8
start_time = time.time()
U, S, V = linalg.svds(A, k=SVDk1)
print("Czas: " + str(time.time() - start_time))
print("Gotowe. Pełny czas: " + str(time.time() - main_start_time) + "ms")
q = input("Wyszukaj: ").split(' ')
matrix, probs = process_query(q, A, mapper, term_by_document)
print("\nNormalizowanie")
result = query(list(matrix), probs, A)
print("\nDokladnosc: " + str(max(result)))
print("Inne wyniki: " + str(prepareOutput(result)))
print("Uzywajac SVD:")
result = querySVD(list(matrix), probs, A, U, S, V, SVDk2)
print("Dokladnosc: " + str(max(result)))
print("Inne wyniki: " + str(prepareOutput(result)))