![Image](images/song_graph.png)
<p align="center"><em>Graph structure</em></p>

# Motivation
With the rise of music streaming services on the internet in the 2010’s, many have moved away from radio stations to streaming services like Spotify and Apple Music. This shift offers more specificity and personalization to users’ listening experiences, especially with the ability to create playlists of whatever songs that they wish. Oftentimes user playlists have a similar genre or theme between each song, and some streaming services like Spotify offer recommendations to expand a user’s existing playlist based on the songs in it. Using Node2vec and GraphSAGE graph neural network methods, we set out to create a recommender system for songs to add to an existing playlist by drawing information from a vast graph of songs we built from playlist co-occurrences (edges between two songs that exist in the same playlist). The result is a personalized song recommender based not only on Spotify’s community of playlist creators, but also the specific features within a song.

# Data
Our song song recommendation system will work with any music dataset that contains a community of users with playlists that they have created. The most popular of these would likely come from Apple Music, Spotify, Youtube, or Amazon as they are by far the most used music streaming services in America (that support playlist creation) as of January 2021. In all the markets Spotify and Youtube contend for the most used, but Youtube is not solely a music streaming service, and they do not release data for public use as readily as Spotify does, so we chose to go with Spotify as our dataset.

In January 2018, Spotify released a vast dataset containing 1 million playlists created by users between January 2010, and October 2017 for the purpose of an online data competition to try to predict subsequent tracks within a playlist. Though the competition is over, we used this dataset of user’s playlists to try to create personalized recommendations for a user’s playlist. Currently we have taken the first ten thousand playlists from this dataset to train our model on, though scaling up to include more playlists (and subsequently songs) is possible, but currently not necessary for us to demonstrate the efficacy of this recommender. 

<p align="center">
  <img src="images/sample_playlist.png" width="50%">
</p>
<p align="center"><em>Sample playlist format</em></p>


Additionally we used Spotify's API to obtain various musical features of songs in order to enrich the recommender to learn from not only the graph structure we created for songs, but also learn from unique aspects of the songs themselves. Some of these features include acousticness, instrumentalness, and danceability as described by Spotify and their algorithmic scoring of such categories.

# Features
### Song Attributes

From these ten thousand playlists, we extracted all of the unique songs, which comes out to around 170,000 unique songs. We then utilized the Spotify developer public API to query information about each of these songs and obtain features for our model. These features include Spotify’s own extracted numerical data from each song, of which we kept the following:

- Danceability (Numerical) - How suitable a track is for dancing.
- Energy (Numerical) - Intensity and activity.
- Loudness (Numerical) - Overall loudness of a track in decibels. 
- Speechiness (Numerical) - Presence of spoken words in a track.
- Acousticness (Numerical) - How acoustic the track is.
- Instrumentalness (Numerical) - How instrumental the track is.
- Liveness (Numerical) - The presence of an audience in the recording.
- Valence (Numerical) - The musical positiveness conveyed by a track.
- Tempo (Numerical) - Estimated tempo in beats per minute.
- Duration (Numerical) - Duration of the song in milliseconds.
- Key (Categorical) - The key that the track is in.
- Mode (Categorical) - Major or minor modality of a track.
- Time Signature (Categorical) - Estimate of time signature.

For our recommender system to successfully provide personalized recommendations, we work under the assumption that when users create playlists manually, they generally will add songs that are similar to each other in some ways. A playlist could be comprised of songs pertaining to a specific genre like dance music or r&b, but it could also reflect a specific mood like happy songs that make you want to dance, or quiet sad songs. So within a playlist, we would expect the measures of the features above to be quite close to each other.

# Graph
The graph we created consists of about 170,000 nodes corresponding to each unique song, and a vast set of edges connecting the songs that appear in a playlist from the first 10,000 playlists we selected. To create an effective recommender, we needed a way to rank the closeness of two songs, so as our aggregate we decided on co-occurrence of songs within playlists as the edges between them with a weight on each edge representing the amount of co-occurrences across all playlists. We chose co-occurrence for our graph because we want to capture node neighborhoods of songs that are alike for our recommender, and we assume that people will create playlists of songs that are at least somewhat alike. We believe this is sufficient for this purpose, but with future optimizations and time to re-create graph structure, trying different methods for graph creation could yield potentially beneficial results. Each node also contains a feature set of the features that are described above. With weighted edges and node features, we would have enough data to create a personalized link prediction problem. Our result was a weighted adjacency matrix with the following measures:
  
<p align="center">
  <img src="images/graph_stats.png" width="50%">
</p>
<p align="center"><em>Graph stats</em></p>

Below you can see how the graph structure is in an image. The co-occurences are counted by edge and the red edge in this case would have a weight = 2 due to co-occurence happening in 2 different playlists. All other black edges would have a weight = 1.
  
<p align="center">
  <img src="images/song_graph.png" width="70%">
</p>
<p align="center"><em>Graph structure</em></p>

# Embeddings
### Node2Vec
One of the earlier graph-based methods is the Node2Vec which uses biased random walks to create low dimensional space representations for nodes. This algorithm aims to preserve node neighborhood networks for the node embeddings and it allows for more accurate classification on nodes because of these neighborhoods. The algorithm utilizes biased 2nd order random walks at the core of its algorithm with p and q tunable parameters to determine the probability of each node subsequent from the original of being visited. Tuning of these parameters allows for the user’s choice of having a more local walk emulating bread-first sampling, or a more explorative walk emulating depth-first sampling. The p parameter determines the probability of a node being revisited right after a step, where a high value makes it less likely that the node is revisited, promoting a depth-first random walk. The q parameter controls the probabilities of stepping to an unvisited node, where a higher value is biased towards local nodes and a smaller value promotes visiting nodes farther from the original.

<p align="center">
  <img src="images/node2vec_explained.png" width="50%">
</p>
<p align="center"><em>Visualization of the node2vec random walk. “t” is the starting node, “v” is the current node in the random walk.</em></p>

### GraphSAGE
GraphSAGE is a framework for node embeddings that separates itself from existing transductive methods that require every node to be present in the training process by proposing an inductive approach to create node embeddings. This is a much more scalable approach that is ideal for large networks that don’t all fit into memory, and that can be continuously updated. The graphSAGE algorithm’s inductive ability comes from its use of neighborhood feature aggregation for a node to create an embedding for that node that will capture information about its neighborhood. Aggregation can be done in a number of ways, the most common being LSTM, mean, pooling aggregator functions. Depending on the aggregator chosen, graphSAGE will capture different information from its neighboring nodes. It is the neighborhood aggregation that makes it so that when a new node is added to the graph, its embedding can be generated from its features and neighboring nodes, rather than having to create new embeddings using the entire graph structure over again.

<p align="center">
  <img src="images/graphSAGE_explained.png" width="70%">
</p>
<p align="center"><em>Visualization of feature aggregation from a node’s neighborhood.</em></p>

# Recommenders
### K-Nearest Neighbors (KNN) on Node2Vec Embeddings
As a baseline model, we built Node2Vec embeddings for songs in the graph based on the weighted connections of co-occurrences in playlists that existed. Using these node embeddings, we fed them into a K-Nearest Neighbor model to find similar embeddings to serve as recommended songs for each song. We explored this model because this is a common method used to encode graph data into a single feature set alongside song features and doesn’t suffer from scalability issues like Graph Convolutional Networks do. It also was able to account for weighted edges that we created in our song graph. After finding similar embeddings for each song, we treated those as predicted edges to evaluate against ground truth to compute precision and recall metrics. We also aggregated the closest embeddings to be incorporated into a recommender. Some drawbacks we expected were performance and accuracy of Node2Vec embeddings in representing neighborhoods as well as runtime issues when computing K-Nearest Neighbor on heavily connected songs with high degrees.

### Link Prediction Ranking on GraphSAGE Embeddings
With the trained embeddings of songs in hand, given a playlist with k seed tracks, for each seed track s and each the candidate track c for recommendation, we will feed the embeddings of s and c into the predictor to compute the predicted scores, which indicates the probability of a edge existing in between. After computing the scores for each seed song, we rank the scores in descending order and prioritize the recommendation of those songs that have the highest probability of a link with the seed tracks. Since there are many possible candidate tracks, we chose the candidates from existing edges and ranked them based on the prediciton scores.

# Results
### Link Prediction Metrics

<p align="center">
  <img src="images/link_pred_metrics.png" width="70%">
</p>
<p align="center"><em>Link Prediction Metrics</em></p>

After running these two models and their respective embeddings, we found that the GraphSAGE embeddings with the MLP predictor gave the highest accuracy. This is as expected, as Node2vec embeddings are quite a bit less complex than the convolutional learning layers that are used in the GraphSAGE embeddings. The high precision entailed that the model was good at predicting edges between songs that should exist for a given song and limited false positives overall. We were more interested in recall as it told us the ratio of predictions out of all edges that do exist between songs. This was important as we believed the strongest recommender would first pull from existing edges that come from a song but aren’t necessarily in a given playlist yet. We saw improvement in both categories and also saw higher AUC scores in GraphSAGE - MLP. The AUC score increase showed better performance in predicting true positives and true negatives over all the data. We can also see this in the ROC curves below that show GraphSAGE - MLP to have a curve closer to the top left than the Node2vec - KNN curve.

# Conclusion

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text
```
