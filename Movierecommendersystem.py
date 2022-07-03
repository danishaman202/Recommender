# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 23:42:39 2022

@author: user
"""

import streamlit as st
import pandas as pd 
from sklearn.decomposition import TruncatedSVD
import numpy as np
column = ['user_id','item_id','rating','timestamp']
data = pd.read_csv('C:/Users/user/Documents/Data Science Projects/Recommender system/ml-100k/u.data', sep = "\t", names = column)
column2 = ['item_id','movie name','release_date', 'video release date','imdb URL', 'unknown', 'action','adventure','animation', 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film noir', 'horror', 'musical', 'mystery', 'romance', 'scifi', 'thriller', 'war', 'western']
movie = pd.read_csv('C:/Users/user/Documents/Data Science Projects/Recommender system/ml-100k/u.item', sep = "|", encoding='latin-1', names = column2)
data_2 = pd.merge(data,movie, on = 'item_id')
crosstab = data_2.pivot_table(values = 'rating', columns = 'movie name', index = 'user_id', fill_value = 0)
X = crosstab.T.values
SVD = TruncatedSVD(n_components = 12, random_state = 17)
result_m = SVD.fit_transform(X)
corrcof = np.corrcoef(result_m)
m = crosstab.columns
m = list(m)

recommender = pd.DataFrame(corrcof,columns = m,index = m)
def recommend(name):
    X = pd.Series(recommender[name].nlargest(5).index)
    X.drop(0,inplace = True)
    return X

def namer(number):
    name = m[number]
    return recommend(name)
    

st.title("Movie Recommender System")
st.write("The List of the movies is:")
st.write(pd.Series(m))
st.write("Please enter the index of the movie that you selected:")
choice = st.number_input("Please enter the number:",min_value= 0, max_value = 1663)
st.write("The most related movies to your input are :")
st.write(namer(choice))