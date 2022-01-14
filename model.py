# Importing required libraries
import pickle
import pandas as pd

# Loading training data
trainingData = pd.read_csv('Dataset/sentimentClassifier_TrainingData.csv')
trainingData.columns = ['index', 'Reviews', 'ProductName']
trainingData.set_index('index', inplace=True)

#Loading tfidf vectorizer
tfidf_Vectorizer = pickle.load(open('Pickle/tfidf_Vectorizer.pkl', 'rb'))

# Loading Sentiment Classifier Model
SentimentClassifier = pickle.load(open('Pickle/SentimentClassifier.pkl', 'rb'))

# Loading Recommender System
RecommenderSystem = pickle.load(open('Pickle/RecommenderSystem.pkl', 'rb'))




### Defining function for getting Optimized/Tuned Recommendations
def getOptimizedRecommendations(username):
    '''Input : Takes username as input
       Output : Returns Top 5 Optimized Recommendations using Recommendation Engine and Sentiment Classifier
       
       Functions Tasks :
       1. Runs Recommendation Engine/System to get Top 20 recommendations.
       2. For each product in Top 20 recommendations, it will fetch all reviews related to it.
       3. For each product, it will classify all reviews as either Positive/Negative after applying tfidf vectorizer.
       4. It will calculate percentage of Positive reviews.
       5. According to percentage of Positive reviews among all reviews, it will sort the Top 20 recommendations
          and return only the Top 5 recommendations.       
    '''
    
    # Creating blank dataframe for storing recommendations
    Recommendations = pd.DataFrame()
    
    # 1. Running Recommendation Engine to get Top 20 Recommendations
    try:
        Recommendations['Top20Recommendations'] = list(
            RecommenderSystem.loc[username].sort_values(ascending=False)[0:20].index)
    except:
        return ('Username is not available in Recommender Engine Database! Please Input the username which is available in Recommender System Database.')
        
    
    # 2. For each product in Top 20 recommendations, fetching all reviews
    # Running loop for each recommended product
    for recommendation in list(Recommendations['Top20Recommendations']):
        # Fetching all reviews for current recommended product
        reviews = trainingData.loc[trainingData.ProductName==recommendation, 'Reviews']
        
        # Creating blank list for storing sentiment of all respective reviews
        sentiment=[]
    
        # 3. For each reviews of current recommended product, vectorizing it and then passing to sentiment classifier
        #    to identify sentiment of respective review
        # Running loop for each review of current recommended product
        for review in reviews:
            
            # Applying tfidf vectorizer for passing as input to sentiment classifier model
            transformedReview = tfidf_Vectorizer.transform(pd.Series(review))
            
            # Predicting sentiment by using Sentiment Classifier with a threshold of 0.6
            predictedProbability = SentimentClassifier.predict_proba(transformedReview).T[1]
            predictedSentiment = 'Positive' if predictedProbability>0.6 else 'Negative'
            
            # Appending sentiment for current review
            sentiment.append(predictedSentiment)
            
        # 4. Calculating Positive Percentage for current recommended product
        positiveSentimentCount = sentiment.count('Positive') 
        positivePercentage = (positiveSentimentCount/len(reviews))*100
        
        # Storing positive percentage for current recommended product in PositivePercentage
        # column of Recommendations dataframe
        Recommendations.loc[Recommendations.Top20Recommendations == recommendation,'positivePercentage']=positivePercentage

    # 5. Sorting Recommendations according to positive percentage in descending order
    Recommendations.sort_values(['positivePercentage', 'Top20Recommendations'], ascending=False, inplace=True)

    # Returning Top 5 Optimized Recommendations
    return list(Recommendations[:5].Top20Recommendations)