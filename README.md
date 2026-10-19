# Fake News Detection using DistilBERT

M.Tech AI project. Classifies a news article as Fake or Real, comparing a TF-IDF + Logistic Regression baseline with a fine-tuned DistilBERT, plus a Streamlit demo app.

## Results (8,980 unseen test articles)

| Model | Accuracy | Total errors |
|---|---|---|
| TF-IDF + Logistic Regression | 97.94% | 185 |
| DistilBERT | 99.78% | 20 |

## Dataset
ISOT Fake News dataset (Kaggle), 44,898 articles. The "(Reuters)" tag in real articles was removed, because it leaks the label and inflates accuracy.

## Run the demo
1. `pip install -r requirements.txt`
2. Train the model with the notebook and save it in a folder named `distilbert_model` next to `app.py` (the model file is not included because of its size).
3. `streamlit run app.py`

## Limitations
Trained on 2016-17 US political news, English only. The model learns writing style, it does not verify facts.

## Tech stack
Python, PyTorch, Hugging Face Transformers, scikit-learn, Streamlit 
