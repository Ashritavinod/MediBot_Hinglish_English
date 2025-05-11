import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class MediBot:
    def __init__(self, faq_path, symptom_path):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.faq_df = pd.read_csv(faq_path)
        self.symptom_df = pd.read_csv(symptom_path)
        self.symptom_dict = dict(zip(self.symptom_df['symptom'], self.symptom_df['advice']))

        self.questions = self.faq_df['question_english'].tolist()
        self.answers = self.faq_df['answer'].tolist()
        self.embeddings = self.model.encode(self.questions)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))

    def get_faq(self, query):
        query_vec = self.model.encode([query])
        D, I = self.index.search(query_vec, 1)
        return self.answers[I[0][0]]

    def get_symptom_advice(self, text):
        for symptom in self.symptom_dict:
            if symptom in text.lower():
                return self.symptom_dict[symptom]
        return None

    def reply(self, user_input):
        symptom_advice = self.get_symptom_advice(user_input)
        if symptom_advice:
            return f"🩺 Symptom Advice:\n{symptom_advice}"
        else:
            return f"📋 FAQ Answer:\n{self.get_faq(user_input)}"