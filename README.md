# 🔍 Fake News Detection System

Binary classification of news articles as real or fake, using BERT fine-tuned with LoRA on 44,000 articles from the LIAR and WELFake datasets.

[![Streamlit App](https://fake-news-detection-nvcn7jnblw5bmqqwvbdewx.streamlit.app/)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Base model | bert-base-uncased |
| Fine-tuning | LoRA via Hugging Face PEFT |
| Training framework | Hugging Face Trainer API |
| Dataset | LIAR + WELFake |
| Frontend | Streamlit |

---

## Results

- **Test Accuracy:** ~93%
- **F1 Score:** ~0.9
- **Training data:** 44,000 articles (LIAR + WELFake)
- **Baseline (TF-IDF + Logistic Regression):** ~82%

---

## Project Structure

```
fake-news-bert/
├── app.py
├── requirements.txt
├── notebooks/
│   └── training.ipynb  
├── .streamlit/
│   └── config.toml
└── README.md
```

## Its Weak Areas

- Satire — formal language from The Onion-style outlets gets misclassified
- Short headlines — too little context for high confidence
- Domain shift — trained on US news data, weaker on translated content
- Works best on inputs of 50+ words

---

## Author

**Himanshu** — B.Tech CSE, MD University
[LinkedIn](https://www.linkedin.com/in/himanshu-906193348/) · [GitHub](https://github.com/himanshu-57)
