from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds as svd
from numpy import diag
import numpy as numpy
import time
import os

__author__ = 'kruczjak'

dir = "text/"
bag_of_words = {}
texts = []


def update_progress(progress, max, time):
    proc = (100 * progress) / max
    print '\r[{0}]{4}% {1}/{2} time: {3}ms'.format('#' * (proc / 5), progress, max - 1, time * 1000, proc),


def prepare(text):
    return text.replace(",", "").replace(".", "").replace(" - ", " ").replace("\'", " ").lower().split()


query = raw_input("Wyraz: ")
k = int(raw_input("Ilosc wynikow: "))

path, dirs, files = os.walk(dir).next()
text_count = len(files) - 1

print "Number of files: " + str(text_count)

main_start_time = start_time = time.time()
print "Rozpoczynam czytanie"
for i in range(text_count):
    file = open(dir + str(i + 1) + ".txt")
    texts.append(set(prepare(file.read())))
    file.close()
    update_progress(i, text_count, time.time() - start_time)

# 2, 3
start_time = time.time()
print "\nTworzenie bag of words"
for i in range(text_count):
    for word in texts[i]:
        if word not in bag_of_words.keys():
            bag_of_words[word] = len(bag_of_words)
    update_progress(i, text_count, time.time() - start_time)


N = len(bag_of_words)
T = len(texts)

# 4
start_time = time.time()
print "\nTworzenie term-by-document matrix"
term_by_document = lil_matrix((N, T))
for i in range(text_count):
    for word in texts[i]:
        term_by_document[bag_of_words[word], i] += 1
    update_progress(i, text_count, time.time() - start_time)
print term_by_document

# 5
start_time = time.time()
print "\nObliczanie inverse document frequency"
idf = [0] * N
for i in range(N):
    idf[i] = term_by_document[i, :].getnnz()
    idf[i] = numpy.log(T / idf[i])
    update_progress(i, N, time.time() - start_time)
print idf
# przemnazanie przez IDF
start_time = time.time()
print "\nNormalizacja z uzyciem inverse document frequency"
for i in range(N):
    for j in range(T):
        term_by_document[i, j] *= idf[i]
    update_progress(i, N, time.time() - start_time)

# 8
start_time = time.time()
print "\nLow-rank approx with SVD"
(U, S, V) = svd(term_by_document, k=k)
temp = U.dot(diag(S)).dot(V)
term_by_document = lil_matrix(temp)
print "Full time: " + str((time.time() - start_time) * 1000)

# 7
start_time = time.time()
print "Normalizacja kazdego wektora do 1"
for i in range(T):
    vec = term_by_document[i, :]
    vec_t = vec.transpose()
    length = vec.dot(vec_t)[0, 0]
    if length != 0:
        term_by_document[i, :] = numpy.multiply(vec, 1 / length)
    update_progress(i, T, time.time() - start_time)

# 6
print "\nWyszukiwanie"
query_vector = lil_matrix((N, 1))
for word in prepare(query):
    query_vector[bag_of_words[word], 0] += 1



for i in range(N):
    try:
        query_vector[i, 0] *= idf[i]
    except KeyError:
        pass


query_length = query_vector.getnnz()
if query_length == 0:
    raise KeyError('Brak wyszukanych slow :(')

query_vector = numpy.multiply(query_vector, 1 / query_length)
result = query_vector.transpose().dot(term_by_document)

print "Wyniki:"
data = result.data
maximum = sorted(data)[(text_count - k):]
for i in range(text_count):
    if result[0, i] in maximum:
        print i
print "Total: " + str((time.time() - main_start_time) * 1000)