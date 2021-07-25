import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
#cosine similarity measures similarity between two different strings
#(A.B)/(|A|.|B|)
def get_recommendations(title,cosine_sim=cosine_sim):
    #get index of given movie
    idx=indices[title]
    #get similarity scores of all other movies
    sim_scores=list(enumerate(cosine_sim[idx]))
    #sort based on similarity score
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
    #get top 10 movies
    sim_scores=sim_scores[1:11]
    movie_indices=[i[0] for i in sim_scores]
    return metadata['title'].iloc[movie_indices]

metadata=pd.read_csv('movies_metadata.csv',low_memory=False)
#remove stop words such as the,and.They have provide no useful data while measuring similarity
tfidf=TfidfVectorizer(stop_words='english')
#fill blank overviews with empty strings
metadata['overview']=metadata['overview'][:100].fillna('')
#construct term Frequency inverse Document frequency vector. Row represent vector and column represents movie.
#tf-idf score is frequency of a word occurring in document,vs the no of documents it occurs in
tfidf_matrix=tfidf.fit_transform(metadata['overview'][:100])
#Calculate cosine similarity
cosine_sim=linear_kernel(tfidf_matrix,tfidf_matrix)
#map of movie and its index
indices=pd.Series(metadata.index,index=metadata['title']).drop_duplicates()


#get_recommendations('Jumanji')

get_recommendations('Toy Story')
