# Emotional Sentiment Analysis



## Project Overview

This project analyzes written text and predicts the dominant emotion expressed in it. The model is trained on emotion-labeled text data and is integrated into a Streamlit user interface for easy interaction.

The application can classify text into the following emotion categories:

- Sadness
- Anger
- Love
- Surprise
- Fear
- Joy

The UI also displays confidence, sentiment polarity, emotion score charts, and downloadable batch results.

## Key Features

- Clean and professional Streamlit interface
- Single text emotion prediction
- CSV batch emotion analysis
- Automatic loading of trained model and vectorizer
- Emotion distribution chart for uploaded datasets
- Confidence and polarity metrics
- Downloadable prediction results
- Sidebar controls for workflow selection and result details

## Technologies Used

- Python
- Streamlit
- Pandas
- Altair
- Scikit-learn
- Joblib
- TF-IDF Vectorization
- Linear Support Vector Classification

## Project Files

```text
Emotional Analysis/
├── app.py
├── model.joblib
├── vectorizer.joblib
├── requirements.txt
├── train.txt
├── page.ipynb
└── README.md
```

## File Description

| File | Description |
| --- | --- |
| `app.py` | Main Streamlit application file |
| `model.joblib` | Trained emotion classification model |
| `vectorizer.joblib` | TF-IDF vectorizer used to transform text |
| `requirements.txt` | Required Python packages |
| `train.txt` | Training dataset used for model development |
| `page.ipynb` | Notebook used for data exploration and model training |
| `README.md` | Project documentation |

## How the Project Works

1. The user enters text manually or uploads a CSV file.
2. The text is transformed using the saved TF-IDF vectorizer.
3. The trained machine learning model predicts the emotion class.
4. The application displays the predicted emotion, confidence, polarity, and charts.
5. For CSV input, the app predicts emotions for all rows and allows downloading the results.

## Emotion Label Mapping

The model uses numeric labels internally. These are mapped to readable emotion names in the application:

| Label | Emotion |
| --- | --- |
| 0 | Sadness |
| 1 | Anger |
| 2 | Love |
| 3 | Surprise |
| 4 | Fear |
| 5 | Joy |

## Installation

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Running the Application

Open a terminal in the project folder and run:

```bash
python -m streamlit run app.py
```

The app will open in the browser at:

```text
http://localhost:8501
```

## Single Text Prediction

Use this mode when you want to test one sentence or paragraph.

Example input:

```text
I feel excited and happy about completing my data science project, but I am also a little nervous about presenting it.
```

The app will show:

- Predicted emotion
- Confidence score
- Polarity score
- Emotion score chart
- Detailed score table

## CSV Batch Prediction

Use this mode when you want to analyze multiple text records at once.

Example CSV format:

```csv
id,name,text
1,Aarav,I feel very happy today
2,Meera,I am nervous about the final presentation
3,Rohan,This result made me angry
```

Steps:

1. Select `CSV batch` from the sidebar.
2. Upload a CSV file.
3. Select the column that contains the text.
4. View the prediction summary and emotion distribution.
5. Download the analyzed results as a CSV file.

The CSV file can contain multiple columns. The important requirement is that it must include at least one column containing text.

## Understanding the Output

### Detected Emotion

The main emotion predicted by the trained model.

### Confidence

Confidence shows how strongly the model supports the predicted emotion. A higher value means the model is more certain about the prediction.

### Polarity

Polarity represents the overall sentiment direction:

| Range | Meaning |
| --- | --- |
| Positive value | Positive emotional tone |
| Around zero | Neutral or balanced tone |
| Negative value | Negative emotional tone |

For example, a polarity value of `-0.65` means the text has a strongly negative emotional tone.

## Model Information

The project uses a text classification pipeline based on:

- TF-IDF vectorization for converting text into numerical features
- LinearSVC model for emotion classification

The saved model and vectorizer are loaded automatically from the same folder as `app.py`.

## Sample Test Inputs

Joy and fear mixed:

```text
I feel excited and happy about completing my data science project, but I am also a little nervous about presenting it in front of everyone.
```

Sadness and anger mixed:

```text
I am really disappointed with the feedback because I worked hard, and now I feel sad, frustrated, and a little hopeless.
```

Surprise:

```text
I was completely surprised by how quickly the model predicted the correct emotion.
```

Love:

```text
I love the way this project turned out because it looks clean, useful, and professional.
```

## Future Improvements

- Add model accuracy and evaluation metrics to the UI
- Add confusion matrix visualization
- Add preprocessing explanation section
- Add support for Excel file upload
- Improve confidence calculation using a probability-based classifier
- Add more advanced NLP models such as Logistic Regression, Random Forest, or Transformer-based models

## Conclusion

This project demonstrates how natural language processing and machine learning can be used to identify emotions from text. The Streamlit interface makes the model easy to use, visually clear, and suitable for project presentation.
