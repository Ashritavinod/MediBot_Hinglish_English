from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import torch
import torch.nn as nn

class MediBot:
    def __init__(self, faq_path, symptom_path, device='cpu'):
        try:
            # Load SentenceTransformer model
            model = SentenceTransformer("all-MiniLM-L6-v2")

            # Check for meta tensors and handle safely
            if any(p.device.type == 'meta' for p in model.parameters()):
                print("Model loaded in 'meta' state. Using to_empty() workaround.")
                model = nn.Module.to_empty(model).to(device)
            else:
                model = model.to(device)

            self.model = model

        except NotImplementedError as e:
            raise RuntimeError("Model loading failed due to meta tensor state. Try clearing cache.") from e

        # Load the FAQ and symptom advice data
        self.faq_df = pd.read_csv(faq_path)
        self.symptom_df = pd.read_csv(symptom_path)

        # Prepare the symptom dictionary
        self.symptom_dict = dict(zip(self.symptom_df['symptom'], self.symptom_df['advice']))

        # Prepare FAQ question embeddings
        self.questions = self.faq_df['question_english'].tolist()
        self.answers = self.faq_df['answer'].tolist()

        self.embeddings = self.model.encode(self.questions, convert_to_numpy=True)

        # Create a FAISS index for similarity search
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def get_faq(self, query):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vec, 1)
        return self.answers[I[0][0]]

    def get_symptom_advice(self, text):
        text_lower = text.lower()
        for symptom in self.symptom_dict:
            if symptom.lower() in text_lower:
                return self.symptom_dict[symptom]
        return None

    def reply(self, user_input):
        symptom_advice = self.get_symptom_advice(user_input)
        if symptom_advice:
            return f"🩺 Symptom Advice:\n{symptom_advice}"
        else:
            return f"📋 FAQ Answer:\n{self.get_faq(user_input)}"
