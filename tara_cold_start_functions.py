
def get_cluster(df, user_id):
    '''df is a dataframe with multiple instances user_id matched to the same cluster'''
    label = df.loc[df.user_id == user_id]['cluster'].mean()
    return label

def get_avg_cluster_rating(df, movie_id, cluster):
    rows = df.loc[(df.movie_id == movie_id) & (df.cluster == cluster)]
    clustered_avg = rows['rating'].mean()
    '''check that movie_id has been rated'''
    if movie_id not in df.movie_id.values:
        print('Error! No info for that movie_id')
    '''check that movie has been rated by members of that cluster'''
    if len(rows) == 0:
        print('Error! Members of that cluster have not rated that movie')
    else:
        return clustered_avg
    
#These are John's functions to personalize the bias

def user_bias(df, user_id):
    return  df.loc[df['user_id'] == user_id, 'rating'].mean() - df['rating'].mean()

def item_bias(df, movie_id):
    return  df.loc[df['movie_id'] == movie_id, 'rating'].mean() - df['rating'].mean()

#get prediction for a movie user has not seen
def tara_get_cold_start(user_id, movie_id):
    '''Load dataframe'''
    df = pd.read_csv('data/movie_rating_user_clusters')
    '''Check that movie is in the dataframe'''
    if movie_id not in df.movie_id.values:
        return 'Error! No info for that movie_id'       
    '''Identify if user_id is clustered. If not, return average reviews for that movie'''
    if user_id not in df.user_id.values:
        predicted_rating = df.loc[df.movie_id == movie_id]['rating'].mean()
    else:
        '''Find cluster'''
        user_cluster = get_cluster(df, user_id)
        '''Find avg cluster rating'''
        cluster_rating = get_avg_cluster_rating(df, movie_id, user_cluster)
        '''Weigh biases'''
        ubias = user_bias(df, user_id)
        ibias = item_bias(df, movie_id)
        predicted_rating = cluster_rating + ubias + ibias
    return predicted_rating