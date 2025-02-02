This project focuses on developing an ECG classification model aimed at identifying various cardiac conditions. 
The model leverages a multi-head attention model to enhance classification, while emphasizing explainability and interpretability to ensure that results can be understood and trusted.

# Features
**High Accuracy:** Utilizing multi-head attention layers showed outperformance over state-of-the-art solutions in terms of classification accuracy (99.96%).
**Explainability:** Implementing explainability is done through visualizing the model's weightsto provide insights into predictions.
**Interpretability:** Highlighting the ECG signal parts contributing in predictions allows users (health professionals) to understand the decision-making process of the model.

# Data
The datasets used for this project are: **_MIT-BIH_** for multiclass classification and **_PTBDB_** for binary classification.
These two datasets are preprocessed. They are retrieved from Kaggle through the following link: https://www.kaggle.com/datasets/shayanfazeli/heartbeat

# Usage
To run the project, download the project zip file or clone the repository and install the required dependencies through the following commands:
* git clone https://github.com/Mus-Kah/ECG_Classification_XAI.git
* pip install requirements.txt
* Enter the project and run the following command line: python main.py
