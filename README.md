# Finetune-FinBERT-StockSentiment  
This project fine-tunes the FinBERT model for sentiment classification on stock news and applies the fine-tuned model to analyze real-time stock news sentiment. The workflow involves training the model on custom datasets and using it to classify stock news articles as `POSITIVE` or `NEGATIVE`.

---

## **Project Overview**

## **Phase 1: Data Preprocess**
Merge data( stock_price and news_data (There are multiple news data for each stock in each day))

### **Phase 2: Fine-Tuning FinBERT**
The first phase fine-tunes the FinBERT model for binary sentiment classification using custom labeled datasets.

#### **Key Steps:**
1. **Load and Preprocess Data:**
   - Load training (`part1.csv`) and testing (`part2.csv`) datasets.
   - Tokenize the text data using a pre-trained tokenizer.
2. **Model Training:**
   - Fine-tune the FinBERT model with specified training parameters, including learning rate, batch size, and number of epochs.
   - Use accuracy as the evaluation metric and save the best-performing model.
3. **Save Fine-Tuned Model:**
   - The fine-tuned model is saved in the `my_awesome_model` directory for later use.

---

### **Phase 3: Real-Time Sentiment Analysis**
The second phase uses the fine-tuned model to classify the sentiment of real-time stock news.

#### **Key Steps:**
1. **Fetch Stock News:**
   - Retrieve recent news articles for a specific stock code using the `akshare` library.
2. **Sentiment Prediction:**
   - Load the fine-tuned model and tokenizer from Phase 1.
   - Classify the sentiment of each news article as `POSITIVE` or `NEGATIVE`.
3. **Summarize Results:**
   - Print the sentiment of each news article alongside the article content.
   - Provide a total count of `POSITIVE` and `NEGATIVE` articles.

---

## **Project Structure**

- **`part1.csv`**: Training dataset.
- **`part2.csv`**: Testing dataset.
- **Data_pre_process.py: Fine-Tuning FinBERT**:
  - Code for loading datasets, fine-tuning the model, and saving the fine-tuned model.
- **Fine_tune.py: Fine-Tuning FinBERT**:
  - Code for loading datasets, fine-tuning the model, and saving the fine-tuned model.
- **Result_fine_tune_model.py: Real-Time Sentiment Analysis**:
  - Code for fetching stock news and applying the fine-tuned model for sentiment classification.

---

## **Outputs**

### **Phase 1: Fine-Tuning Results**
- Fine-tuned model saved in the `my_awesome_model` directory.

### **Phase 2: Sentiment Analysis Results**
- Sentiment classification (`POSITIVE` or `NEGATIVE`) for each news article.
- Total count of `POSITIVE` and `NEGATIVE` articles.

---
