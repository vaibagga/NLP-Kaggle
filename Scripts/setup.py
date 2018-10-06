'''
from distutils.core import setup
from Cython.Build import cythonize
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
from distutils.core import setup
from Cython.Build import cythonize

os.chdir('C:\\Users\\lenovo\\Desktop\\All\\Docs\\Project Paper\\Data')
embeddings = pd.read_csv('Ratings_Warriner_et_al.csv')
embeddings = embeddings[['Unnamed: 0','Word','V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']]

def wordEmbeddings(word):
    return np.array(embeddings.loc[embeddings['Word'] == word].drop(['Word', 'Unnamed: 0'], axis = 1))

def cosineSimilarity(e1, e2):
    e1 = wordEmbeddings(word1)
    e2 = wordEmbeddings(word2)
    return np.dot(e1, e2.T)/(np.linalg.norm(e1) * np.linalg.norm(e2))

setup(
    ext_modules=cythonize("cosineSimilarity.pyx")
)
'''
print("Hello World")
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("setup.py")
)