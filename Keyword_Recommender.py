#Added some more filters while sorting such as Genre,Keyword and Credits. 
#System will be based on the top 3 actors, director,genre along with movie plot keyword
import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_director(x):
    #return director name
    for i in x:
        if i['job']=='Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x,list):
        names=[i['name'] for i in x]
        if len(names)>3:
            names=names[:3]
        return names
    return []

def clean_data(x):
    if isinstance(x,list):
        return [str.lower(i.replace(" ","")) for i in x]
    else:
        if isinstance(x,str):
            return str.lower(x.replace(" ",""))
        else:
            return ''
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

def get_recommendations(title,cosine_sim2=cosine_sim2):
    idx=indices[title]
    sim_scores=list(enumerate(cosine_sim2[idx]))
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
    sim_scores=sim_scores[1:11]
    movie_indices=[i[0] for i in sim_scores]
    return metadata['title'].iloc[movie_indices]

metadata=pd.read_csv('movies_metadata.csv',low_memory=False)
credits=pd.read_csv('credits.csv')
keywords=pd.read_csv('keywords.csv')
#dropping bad data present in csv file
metadata=metadata.drop([19730,29503,35587])
#convert all ids to int
keywords['id']=keywords['id'].astype('int')
credits['id']=credits['id'].astype('int')
metadata['id']=metadata['id'].astype('int')
#add credits and keywords into metadata
metadata=metadata.merge(credits,on='id')
metadata=metadata.merge(keywords,on='id')
features=['cast','crew','keywords','genres']
for feature in features:
    #convert data into acceptable form
    metadata[feature]=metadata[feature].apply(literal_eval)
#get names of directors    
metadata['director']=metadata['crew'].apply(get_director)
features=['cast','keywords','genres']
for feature in features:
    #get top 3 actors,keywords,genres
    metadata[feature]=metadata[feature].apply(get_list)

    features=['cast','keywords','director','genres']
for feature in features:
    #remove spaces in names, A B, A C will be same name if spaces arent removed
    metadata[feature]=metadata[feature].apply(clean_data)
#create soup function joins all required column by  space    
metadata['soup']=metadata.apply(create_soup,axis=1)
#apply cosine similarity to the created list
count=CountVectorizer(stop_words='english')
count_matrix=count.fit_transform(metadata['soup'][:100])

cosine_sim2=cosine_similarity(count_matrix,count_matrix)
metadata=metadata.reset_index()

indices=pd.Series(metadata.index,index=metadata['title'])

get_recommendations('Toy Story',cosine_sim2)
