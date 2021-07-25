import pandas as pd

metadata=pd.read_csv('movies_metadata.csv',low_memory=False)
#mean rating across all movies
C=metadata['vote_average'].mean()
#movie as to have more votes than 90% of the movies on the list
m=metadata['vote_count'].quantile(0.90)
#IMDB Weighted Rating Formula
#WeightedRating(WR)=(v/v+m*R)+(m/v+m*C)
#v=no of votes
#m=min no of votes required to be in chart
#r=average rating of movie
#c mean vote across whole report
def weighted_rating(x,m=m,C=C):
    v=x['vote_count']
    R=x['vote_average']
    return (v/(v+m)*R)+(m/(m+v)*C)

q_movies=metadata.copy().loc[metadata['vote_count']>=m]
#add rating to qmovies 
q_movies['score']=q_movies.apply(weighted_rating,axis=1)
#sort movies according to score
q_movies=q_movies.sort_values('score',ascending=False)
#print q movies
q_movies[['title','vote_count','vote_average','score']].head(10)
