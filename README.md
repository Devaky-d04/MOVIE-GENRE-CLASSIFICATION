# MOVIE-GENRE-CLASSIFICATION
The project focuses on classifying movies into multiple genres using plot summaries. The dataset includes movies with associated titles, plot overviews, and genres. Since each movie can belong to multiple genres, this is a multi-label classification problem, where the goal is to predict a set of labels for each movie.

Key Features:
1.Data Preprocessing:
  Cleans and preprocesses movie plot summaries and genre labels.
2.Feature Extraction:
  Uses TF-IDF vectorization to transform text data into numerical features.
3.Modeling: 
  Trains models using:
    Naive Bayes
    Logistic Regression
    XGBoost
4.Evaluation:
  Measures model performance with accuracy, classification reports, confusion matrices, and precision-recall/ROC curves.
