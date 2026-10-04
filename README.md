# Chinese Weibo Sentiment Analysis

## ML Mini-Project — UE24CS352A

### Team Members
- Mayukh Pravin — PES2UG24CS271
- Medha Singh — PES2UG24CS274

## Project Overview

This project implements Chinese social-media sentiment analysis using
Machine Learning and Deep Learning techniques.

The project is motivated by research on sentiment analysis of Chinese
Weibo posts concerning attitudes toward COVID-19 vaccination.

The main sentiment classifier is trained on a publicly available Chinese
Weibo sentiment dataset and is subsequently applied to COVID-19/vaccine-
related posts as an application-level analysis.

## Models Used

- Naive Bayes
- Support Vector Machine (SVM)
- Bidirectional LSTM (Bi-LSTM)

## Dataset

The project uses the `weibo_senti_100k` Chinese Weibo sentiment dataset.

For computational efficiency, a stratified subset of 30,000 posts was
used for the main experiments.

COVID-19/vaccine-related posts were additionally identified using
keyword-based filtering for application-level analysis.

## Preprocessing

The preprocessing pipeline includes:

- URL removal
- Mention removal
- Chinese text segmentation using Jieba
- Removal of very short tokens
- TF-IDF feature extraction for traditional ML models
- Tokenization and padding for the Bi-LSTM model

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Naive Bayes | 79.78% | 79.29% | 80.63% | 79.95% |
| SVM | 81.08% | 80.87% | 81.43% | 81.15% |
| Bi-LSTM | 81.48% | 80.90% | 82.43% | 81.66% |

The Bi-LSTM achieved the best overall performance among the three
evaluated models.

## Repository Contents

- `ML_MiniProject_Chinese_Vaccine_Sentiment.ipynb` — complete project
  implementation and experiments.

## How to Run

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Upload the `weibo_senti_100k.csv` dataset when prompted.
3. Install the required Python libraries.
4. Run the notebook cells sequentially from top to bottom.
5. The notebook trains the models, evaluates their performance, and
   demonstrates sentiment prediction on new text.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Jieba
- Matplotlib
- Google Colab

## Application

The trained sentiment model is demonstrated on a small set of Chinese
Weibo posts containing COVID-19/vaccine-related keywords.

These posts are used as an application-level demonstration and are not
treated as manually labelled ground truth for model evaluation.
