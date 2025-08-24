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
* Enter the project and run one of the following command lines: 
  * For training without considering patient-level validation:
    * For straightforward training: python .src\exe\normal\main_straightforward.py
    * For training with cross-validation: python main_cv.py .src\exe\normal\main_cross_validation.py
  * For training considering patient-level according to different configurations:
    * For training the transformer without auto-encoder-based processing: python .src\exe\ablation\ablation\main_without_autoencoder.py
    * For training with auto-encoder and with different classification models: python .src\exe\ablation\main.py

# Considerations
* The **main.py** file contains runs all the configurations stated in the study. You can change within it the dataset used for learning, select the classification model, and customize the hyperparameters.

* The default usage is to run main.py on the MIT-BIH dataset using the transformer-based model. This model can be selected in the following line:

  *model_name = None*
  * Choose:
    * **'cnn'** for Convlutional Neural Network model
    * **'lstm'** for Long-Short Term Memory model
    * **'mlp'** for Multi-Layer Perceptron model
    * **'transformer'**, **None**, or simply delete the argument for the default model, which is *Multi-head attention model*.