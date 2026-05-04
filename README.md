# nypd-civilian-complaints-EDA
- Exploratory data analysis and baseline modeling of https://www.propublica.org/datastore/dataset/civilian-complaints-against-new-york-city-police-officers
- Putting to practice concepts from https://dsc80.com/

## Deep neural network baseline
- Run `python dnn_model.py` to train a multi-layer perceptron regressor that predicts `days_taken` from complaint demographics.
- The model uses one-hot encoding for categorical fields and scaling for numeric features in a single sklearn pipeline.
