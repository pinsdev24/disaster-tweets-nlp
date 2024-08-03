# Disaster Tweets NLP Project

## Overview

This project involves building a Natural Language Processing (NLP) model to classify tweets as real disaster tweets or not. The goal is to help emergency responders and analysts quickly identify real disaster-related information from social media.

## Dataset

The dataset used in this project consists of tweets labeled as either real disasters or not. The data is available in a CSV format and contains the following columns:
- `id`: Unique identifier for each tweet
- `text`: The text content of the tweet
- `location`: Location where the tweet was posted (if available)
- `keyword`: A keyword from the tweet (if available)
- `target`: Binary value indicating if the tweet is a real disaster (1) or not (0)

For more details, refer to the `disaster.pdf` document in the repository.

## Project Structure

- `realORnot.ipynb`: Jupyter Notebook containing the entire analysis, model training, and evaluation process.
- `disaster.pdf`: Document providing a detailed description of the dataset.

## Getting Started

### Prerequisites

To run the code in this repository, you need to have the following installed:
- Python 3.6 or higher
- Jupyter Notebook
- Required Python libraries: `pandas`, `numpy`, `sklearn`, `nltk`, `matplotlib`, `seaborn`

You can install the required libraries using the following command:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

### Running the Notebook

1. Clone the repository to your local machine:

```bash
git clone https://github.com/pinsdev24/disaster-tweets-nlp.git
```

2. Navigate to the project directory:

```bash
cd disaster-tweets-nlp
```

3. Open the Jupyter Notebook:

```bash
jupyter notebook realORnot.ipynb
```

4. Execute the cells in the notebook to reproduce the results.

## Project Steps

### 1. Data Preprocessing

- **Loading the Data**: Import the dataset and explore its structure and contents.
- **Cleaning the Data**: Handle missing values, remove irrelevant characters, and preprocess the text for analysis.
- **Text Normalization**: Tokenization, lowercasing, removal of stop words, and stemming/lemmatization.

### 2. Exploratory Data Analysis (EDA)

- Visualize the distribution of real vs. not real disaster tweets.
- Analyze the frequency of keywords and locations.
- Word cloud generation for a visual representation of the most common words.

### 3. Feature Engineering

- Create new features from the text data, such as the length of the tweet, the number of hashtags, mentions, and URLs.
- Use Term Frequency-Inverse Document Frequency (TF-IDF) for text vectorization.

### 4. Model Building

- Split the data into training and testing sets.
- Train multiple models: SVM and FNN for text classification
- Evaluate the models using metrics such as accuracy, precision, recall, and F1-score.

### 5. Model Evaluation and Selection

- Compare the performance of different models.
- Select the best model based on evaluation metrics.

## Results

The final model achieves an accuracy of 90% on the test set.

## Conclusion

This project demonstrates the process of building a robust NLP model for classifying disaster-related tweets. The model can be further improved by experimenting with more advanced techniques and larger datasets.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. Contributions, issues, and feature requests are welcome!