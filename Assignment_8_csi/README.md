# 📄 Assignment 8 CSI — RAG Loan Approval Chatbot

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** chatbot for answering questions related to **loan approvals**.

✅ **Live Demo:** [Open App](https://imageclassificationcsi25internthirdyear-no5uzjbcfhrctjj9yttln3.streamlit.app/)

---

## 🚀 How it works

- Uses **FAISS** to retrieve relevant loan policy sentences.
- Uses **DistilBERT** to embed queries and documents.
- Generates answers using **Flan-T5**.
- Runs interactively on **Streamlit Cloud**.

---

## ✅ Example Questions

Try asking:
- *Who can get a loan approved with good credit score?*
- *Does high income help with loan approval?*
- *What happens if credit history is bad?*

---

## ✨ Tech Stack

- **Python**
- **Hugging Face Transformers**
- **FAISS**
- **Streamlit**

---

##Resource
https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction?select=Training+Dataset.csv

## 📸 Screenshot

![RAG Chatbot Demo](my_rag_chatbot_csi/rag_chatbot_demo.png)

---

## 🟢 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Hardika-Makwana/Image_Classification_CSI25_Intern_thirdYear.git

# Go to project folder
cd Image_Classification_CSI25_Intern_thirdYear/Assignment_8_csi

# (Optional) Create virtual env & activate it

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
