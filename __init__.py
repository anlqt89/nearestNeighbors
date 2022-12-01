from pyresparser import ResumeParser
import os
from docx import Document
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from ftfy import fix_text
from sklearn.neighbors import NearestNeighbors
from resume_parser import resumeparse
import json
# from types import MethodDescriptorType
# import flask
# from flask import jsonify, request
# from flask import json
# from flask_cors import CORS

# app = flask.Flask(__name__)
# app.config["DEBUG"] = True
# CORS(app)

# # http://127.0.0.1:5000/ai?fn=%22resume.csv%22
# @app.route('/ai', methods=['GET'])

# def ai(filename):
#     return jsonify(run_ai(filename))

# app.run()

def run_ai(filename):
  stop_words  = set(stopwords.words('english'))

  #get job decription list
  df =pd.read_csv('jobList.csv') 
  df['test']=df['Job_Description'].apply(lambda x: ' '.join([word for word in str(x).split() if len(word)>2 and word not in (stop_words)]))


  #read resume
  stop_words = set(stopwords.words('english'))
  punctuations = string.punctuation
  # ---------------------------------------------
  # Parse resume:

  #combine skills
  data = resumeparse.read_file(filename)
  skills=[]
  skills.append(' '.join(word for word in data))

  #Begin AI
  def ngrams(string, n=3):
      string = fix_text(string) # fix text
      string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
      string = string.lower()
      chars_to_remove = [")","(",".","|","[","]","{","}","'"]
      rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
      string = re.sub(rx, '', string)
      string = string.replace('&', 'and')
      string = string.replace(',', ' ')
      string = string.replace('-', ' ')
      string = string.title() # normalise case - capital at start of each word
      string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
      string = ' '+ string +' ' # pad names for ngrams...
      string = re.sub(r'[,-./]|\sBD',r'', string)
      ngrams = zip(*[string[i:] for i in range(n)])
      return [''.join(ngram) for ngram in ngrams]


  #convert to 2D dimension
  vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
  tfidf = vectorizer.fit_transform(skills)

  #base line for skills
  nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
  test = (df['test'].values.astype('U'))

  #plot job description 
  def getNearestN(query):
    queryTFIDF_ = vectorizer.transform(query)
    distances, indices = nbrs.kneighbors(queryTFIDF_)
    return distances, indices

  distances, indices = getNearestN(test) #distances: distance between skills and jobs_desc, indices: positions of jobs_desc

  #round up the matching score
  test = list(test) 
  matches = []

  for i,j in enumerate(indices):
      dist=round(distances[i][0],2)
      temp = [dist]
      matches.append(temp)
      
  matches = pd.DataFrame(matches, columns=['Match confidence'])
  df['match']=matches['Match confidence']
  df1=df.sort_values('match')
  df1 = df1[['match', 'url', 'Position', 'Company', 'Location']].head(10).reset_index()
  
  finFile = pd.DataFrame(df1)
  res = json.dumps(finFile.to_dict('records'))
  
  return res