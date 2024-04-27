# COVID-19 Tweet Sentiment Analysis

During the COVID-19 pandemic, social media platforms, especially Twitter, became primary sources for information dissemination. However, the abundance of information led to the spread of misinformation and panic. Our project aims to address this challenge by accurately categorizing tweets into positive, negative, or neutral sentiments.

## Key Features
- **Dataset:** Sourced from Kaggle's COVID-19 NLP Text Classification Dataset.
- **Preprocessing:** Label simplification, data cleaning, lemmatization, and encoding.
- **Imbalance Handling:** Utilized RandomOverSampler.
- **Models Evaluated:** Logistic Regression, SVM, Naive Bayes, XGBoost, Random Forest, Bert, and Electra.
- **Model Accuracy:** Ranges from 72% to 89%.

## Getting Started
1. **Download Dataset:** Download the dataset from [Kaggle](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification/data).
2. **Clone the Repository:** `git clone [repository_url]`
3. **Upload Files:** Upload the downloaded dataset and Jupyter notebook file to the repository.
4. **Explore the Dataset:** Review the Jupyter notebook for dataset exploration and preprocessing techniques.
5. **Train Models:** Utilize provided scripts to train and evaluate machine learning models.
6. **Contributions:** Contributions and feedback from the open-source community are welcome!

## Accuracy Table

| Model                   | Accuracy Rate |
|-------------------------|---------------|
| Logistic Regression     | 79%           |
| Support Vector Machines | 80%           |
| Naive Bayes             | 72%           |
| XGBoost                 | 74%           |
| Random Forest           | 75%           |
| Bert                    | 86%           |
| Electra                 | 89%           |

Note: The models Bert and Electra were trained using the small versions with only 3 epochs. Further training may improve accuracy.

## Explanation of Models

1. **Logistic Regression:**
   - Logistic Regression is a linear model commonly used for multi-class classification tasks.
   - It predicts the probability that a given input belongs to each class.
   - Despite its simplicity, it can perform well in certain scenarios and serves as a baseline model for text classification tasks.

2. **Support Vector Machines (SVM):**
   - Support Vector Machines are versatile supervised learning models capable of performing multi-class classification tasks.
   - They find the hyperplane that best separates classes in a high-dimensional space.
   - SVMs are effective in handling high-dimensional data and can capture complex relationships between features.

3. **Naive Bayes:**
   - Naive Bayes is a probabilistic classifier based on Bayes' theorem with an assumption of independence between features.
   - Despite its simplicity, Naive Bayes can perform well on text classification tasks.
   - It is particularly efficient for large datasets and works well with high-dimensional data.

4. **XGBoost (Extreme Gradient Boosting):**
   - XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.
   - It builds multiple decision trees sequentially, each correcting errors of its predecessors.
   - XGBoost is known for its scalability, efficiency, and accuracy, making it a popular choice for various machine learning tasks.

5. **Random Forest:**
   - Random Forest is an ensemble learning method based on decision trees.
   - It constructs a multitude of decision trees during training and outputs the mode of the classes (classification) or the mean prediction (regression) of the individual trees.
   - Random Forest is robust to overfitting and works well with high-dimensional data.

6. **BERT (Bidirectional Encoder Representations from Transformers):**
   - BERT is a pre-trained natural language processing model developed by Google.
   - It utilizes bidirectional transformers to understand the context of words in a sentence.
   - BERT has achieved state-of-the-art results on various NLP tasks due to its ability to capture deep contextual relationships.

7. **Electra (Efficiently Learning an Encoder that Classifies Token Replacements Accurately):**
   - Electra is another pre-trained transformer-based NLP model designed for efficient training.
   - It improves upon the efficiency of BERT by focusing on the discriminator part of the model.
   - Electra achieves competitive performance while requiring less computational resources compared to BERT.

Note: Each model has its strengths and weaknesses, and the choice of model depends on the specific characteristics of the dataset and the task at hand.

## Explanation of TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF is a numerical statistic used to reflect the importance of a word in a document relative to a collection of documents. It is commonly used in information retrieval and text mining.

1. **Term Frequency (TF):**
   - Term Frequency measures how frequently a term appears in a document.
   - It is calculated as the number of times a term appears in a document divided by the total number of terms in the document.
   - TF increases with the number of occurrences of a term within a document.

2. **Inverse Document Frequency (IDF):**
   - Inverse Document Frequency measures the importance of a term across a collection of documents.
   - It is calculated as the logarithm of the total number of documents divided by the number of documents containing the term.
   - IDF decreases with the number of documents containing the term, indicating that common terms are less important.

3. **TF-IDF Calculation:**
   - TF-IDF is calculated as the product of TF and IDF.
   - It combines the local importance of a term (TF) with its global importance (IDF) across a collection of documents.
   - Words with high TF-IDF scores are important within a document but relatively rare across the entire collection, making them good candidates for distinguishing between documents.

TF-IDF is commonly used for feature extraction and vectorization in text mining and natural language processing tasks, including document classification, clustering, and information retrieval.


## Acknowledgments
- Kaggle and the contributors to the COVID-19 NLP Text Classification Dataset.
- Open-source community for valuable resources and insights.

