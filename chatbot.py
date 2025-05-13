import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
import torch.nn as nn

class MediBot:
    def __init__(self, faq_path, symptom_path, device='cpu'):
        # Load model safely, handling potential meta tensor issues
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Check if model parameters are on 'meta' device
        if any(p.device.type == 'meta' for p in model.parameters()):
            model = nn.Module.to_empty(model)  # creates structure without data
            model = model.to(device)
        else:
            model = model.to(device)

        self.model = model

        # Load data
        self.faq_df = pd.read_csv(faq_path)
        self.symptom_df = pd.read_csv(symptom_path)
        self.symptom_dict = dict(zip(self.symptom_df['symptom'], self.symptom_df['advice']))

        # Prepare embeddings
        self.questions = self.faq_df['question_english'].tolist()
        self.answers = self.faq_df['answer'].tolist()
        self.embeddings = self.model.encode(self.questions, convert_to_numpy=True)

        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))

    def get_faq(self, query):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vec, 1)
        return self.answers[I[0][0]]

    def get_symptom_advice(self, text):
        for symptom in self.symptom_dict:
            if symptom.lower() in text.lower():
                return self.symptom_dict[symptom]
        return None

    def reply(self, user_input):
        symptom_advice = self.get_symptom_advice(user_input)
        if symptom_advice:
            return f"🩺 Symptom Advice:\n{symptom_advice}"
        else:
            return f"📋 FAQ Answer:\n{self.get_faq(user_input)}"
