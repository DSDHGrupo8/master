# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:43:35 2019

@author: ASEBA1
"""

from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from itertools import product as iproduct
from functools import reduce
from operator import mul
from collections import Counter
from itertools import chain
import numpy as np


class PolyDictVectorizer(DictVectorizer):

    def __init__(self, degree=2, sparse=True, num_types=[float, np.float64]):
        self.degree = degree
        self.num_types = num_types
        super().__init__(sparse=sparse)

    def fit(self, X, y=None):
        X = [encode(x, self.degree, self.num_types) for x in X]
        return super().fit(X, y)

    def _transform(self, X, fitting):
        X = [encode(x, self.degree, self.num_types) for x in X]
        return super()._transform(X, fitting)

    
class PolyFeatureHasher(FeatureHasher):

    def __init__(self, degree=2, n_features=2**20, num_types=[float, np.float64]):
        self.degree = degree
        self.num_types = num_types
        super().__init__(n_features=n_features)

    def transform(self, X):
        X = [encode(x, self.degree, self.num_types) for x in X]
        return super().transform(X)


def product(iterable, start=1):
    return reduce(mul, iterable, start)


def encode(dic, degree, num_types):
    dic = {k if type(v) in num_types else str(str(k) + "=" + str(v)):
           float(v) if type(v) in num_types else 1
           for k, v in dic.items()}
    
    #print("dic:", dic)
    aux_dic={}

    dic_keys = list(dic.keys())
    for deg in range(2, degree + 1):
        for term_keys in iproduct(dic_keys, repeat=deg):
            term_names, term_facts = [], []
            for k, n in Counter(term_keys).items():
                v = dic[k]
                if type(v) is int and n > 1:
                    break
                #term_names.append(k if n == 1 else str(str(k) + "^" + str(n)))
                #if (type(v) in num_types):
                if (n==1):
                    term_names.append(k)
                else:
                    aux_str=str(k) + "^" + str(n)
                    term_names.append(aux_str)
                        
                    #print("v:", v)
                    #print("n:", n)
                    term_facts.append(v**n)
            else:  # No dummy feature was included more than once
                dic['*'.join(sorted(term_names))] = product(term_facts)
    
    output_dic=dict(chain.from_iterable(d.items() for d in (aux_dic,dic)))
    return output_dic