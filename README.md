# ANN Binary Classification – Personality Prediction

An Artificial Neural Network (ANN) project that performs **binary personality classification** from text. The model analyzes short text statements and predicts whether the personality is **Introvert** or **Extrovert**.

## 📌 Project Overview

This project demonstrates a complete Natural Language Processing (NLP) and Deep Learning workflow:

**Text Data → Data Cleaning → Target Encoding → Train/Test Split → TF-IDF → ANN → Prediction → Evaluation**

The input is a text statement, while the target variable is the personality class.

- **Input:** Text
- **Target:** Personality
- **Classes:** Introvert / Extrovert
- **Task:** Binary Classification
- **Model:** Artificial Neural Network

## 📊 Dataset

The project uses `personality_dataset_2000.csv`.

The dataset contains **2,000 records** and **2 columns**:

| Column | Description |
|---|---|
| `Text` | Text statement describing a person's preference, behavior, or feeling |
| `Personality` | Target class: `Introvert` or `Extrovert` |

### Class Distribution

The dataset is balanced:

- **Extrovert:** 1,000 samples
- **Introvert:** 1,000 samples

The notebook checks for missing values and duplicate rows. In the executed results, there were **0 missing values** and **0 duplicate rows**, so the dataset remained at 2,000 records after cleaning.

## 🧹 Data Preprocessing

The following preprocessing steps are implemented:

1. Load the CSV dataset using Pandas.
2. Inspect dataset shape, columns, data types, missing values, and duplicates.
3. Remove duplicate rows.
4. Remove rows where `Text` or `Personality` is missing.
5. Separate the text feature (`X`) and target (`y`).
6. Encode the target labels:
   - `Introvert → 0`
   - `Extrovert → 1`
7. Split the dataset into training and testing sets using an **80/20 split** with stratification.

### Train/Test Split

- Training samples: **1,600**
- Testing samples: **400**

## 🔤 TF-IDF Text Vectorization

Since the ANN requires numerical input, the text is converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

The vectorizer is configured with:

- `max_features=5000`
- `lowercase=True`
- `stop_words='english'`

The vocabulary is learned only from the training data using `fit_transform()`, while the test data is transformed using the same fitted vectorizer. This helps maintain a consistent feature representation between training and testing.

In the executed notebook, TF-IDF produced:

- Training matrix: **(1600, 1002)**
- Testing matrix: **(400, 1002)**

The sparse TF-IDF matrices are converted to dense arrays before being passed to the ANN.

## 🧠 ANN Model Architecture

The project uses a feed-forward Artificial Neural Network implemented with TensorFlow/Keras.

```text
Input Layer
     ↓
Dense Layer – 128 neurons, ReLU
     ↓
Dropout – 20%
     ↓
Dense Layer – 64 neurons, ReLU
     ↓
Dropout – 20%
     ↓
Dense Layer – 32 neurons, ReLU
     ↓
Output Layer – 1 neuron, Sigmoid
```

### Model Configuration

| Component | Configuration |
|---|---|
| Hidden Layer 1 | 128 neurons, ReLU |
| Dropout | 0.2 |
| Hidden Layer 2 | 64 neurons, ReLU |
| Dropout | 0.2 |
| Hidden Layer 3 | 32 neurons, ReLU |
| Output Layer | 1 neuron, Sigmoid |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Metric | Accuracy |
| Epochs | 30 |
| Batch Size | 32 |
| Validation Split | 20% of training data |

The executed model contains **138,753 trainable parameters**.

## 🎯 Why Sigmoid?

This is a binary classification problem with two possible classes. The sigmoid output produces a probability between 0 and 1.

- Probability `< 0.5` → **Introvert**
- Probability `>= 0.5` → **Extrovert**

## 🏋️ Model Training

The ANN is trained using the TF-IDF training features. A **20% validation split** is taken from the training data to monitor validation performance during the 30 training epochs.

Training history is stored and used to visualize:

- Training Accuracy vs Validation Accuracy
- Training Loss vs Validation Loss

## 📈 Model Evaluation

The model is evaluated on the unseen test set using:

### Accuracy

Measures the proportion of correctly classified test samples.

The notebook calculates accuracy using `accuracy_score()` and reports the result as a percentage.

### Classification Report

The classification report provides:

- Precision
- Recall
- F1-score
- Support

for both **Introvert** and **Extrovert** classes.

### Confusion Matrix

A confusion matrix is generated to show the number of:

- Introverts correctly predicted as Introverts
- Introverts incorrectly predicted as Extroverts
- Extroverts incorrectly predicted as Introverts
- Extroverts correctly predicted as Extroverts

The project also visualizes the confusion matrix using Matplotlib.

## 📊 Training Visualizations

The notebook generates two training graphs.

### 1. Accuracy Curve

Compares training accuracy with validation accuracy across epochs. This helps observe learning behavior and potential overfitting.

### 2. Loss Curve

Compares training loss with validation loss across epochs. This helps understand how the model's optimization changes during training.

## 🔮 External Text Prediction

The project also supports prediction on a completely new text statement.

Example:

```text
I enjoy spending time with many people and talking to everyone
```

The new text is transformed using the **same trained TF-IDF vectorizer**, passed to the trained ANN, and converted into a personality prediction.

The notebook displays:

- Input text
- Extrovert probability
- Predicted personality

## 🛠️ Technologies Used

- **Python**
- **Pandas** – data loading and preprocessing
- **NumPy** – numerical operations
- **Scikit-learn** – train/test split, TF-IDF, and evaluation metrics
- **TensorFlow / Keras** – ANN model development and training
- **Matplotlib** – visualization
- **Jupyter Notebook** – development and experimentation

## 📁 Repository Structure

```text
ANN-Binary-Classification/
│
├── ANN-BINARY.ipynb       # Complete implementation and analysis
└── README.md              # Project documentation
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/mahi4221/ANN-Binary-Classification.git
cd ANN-Binary-Classification
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow jupyter
```

### 3. Open the notebook

```bash
jupyter notebook ANN-BINARY.ipynb
```

### 4. Dataset Path

Update the `file_path` variable in the notebook to point to your local copy of `personality_dataset_2000.csv`.

Example:

```python
file_path = r"path/to/personality_dataset_2000.csv"
```

## 🔄 End-to-End Workflow

```text
Personality Text Dataset
          ↓
Data Exploration
          ↓
Remove Duplicates / Missing Values
          ↓
Encode Personality Labels
          ↓
80/20 Train-Test Split
          ↓
TF-IDF Vectorization
          ↓
ANN Model
          ↓
Training + Validation
          ↓
Test Prediction
          ↓
Accuracy + Classification Report
          ↓
Confusion Matrix
          ↓
External Text Prediction
```

## 💡 Key Learning Outcomes

Through this project, the following concepts are demonstrated:

- Binary classification using neural networks
- Natural Language Processing with TF-IDF
- Text-to-numerical feature conversion
- Target label encoding
- Stratified train/test splitting
- ANN architecture design
- ReLU and sigmoid activation functions
- Dropout regularization
- Adam optimization
- Binary crossentropy loss
- Model validation and evaluation
- Confusion matrix interpretation
- Classification metrics
- Training/validation curve analysis
- Prediction on unseen text

## ⚠️ Notes

The dataset is relatively small and consists of short text statements. Therefore, the reported test accuracy should be interpreted in the context of this particular dataset. A very high accuracy does not necessarily mean the model will generalize equally well to real-world personality prediction data.

## 👨‍💻 Author

**Maruthi Bonela**

B.Tech – Computer Science Engineering (AI & ML)

GitHub: https://github.com/mahi4221

---

⭐ If you find this project useful, feel free to explore the notebook and experiment with different TF-IDF settings, ANN architectures, and hyperparameters.