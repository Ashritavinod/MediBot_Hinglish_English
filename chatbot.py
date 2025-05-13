from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import torch

class MediBot:
    def __init__(self, faq_path, symptom_path, device='cpu'):
        self.device = device

        # Load SentenceTransformer model safely
        self.model = self.load_model("all-MiniLM-L6-v2")

        # Load FAQ and symptom data
        self.faq_df = pd.read_csv(faq_path)
        self.symptom_df = pd.read_csv(symptom_path)

        # Build symptom dictionary
        self.symptom_dict = dict(zip(self.symptom_df['symptom'], self.symptom_df['advice']))

        # Prepare FAQ data
        self.questions = self.faq_df['question_english'].tolist()
        self.answers = self.faq_df['answer'].tolist()

        # Encode questions
        self.embeddings = self.model.encode(self.questions, convert_to_numpy=True)

        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def load_model(self, model_name):
        try:
            model = SentenceTransformer(model_name)
            # Handle meta tensors (edge case in custom transformers or colab bugs)
            if any(p.device.type == 'meta' for p in model.parameters()):
                raise RuntimeError("Model loaded in 'meta' state. Try clearing torch cache.")
            return model.to(self.device)
        except NotImplementedError as e:
            raise RuntimeError("Model loading failed due to meta tensor state.") from e

    def get_faq(self, query):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, 1)
        return self.answers[indices[0][0]]

    def get_symptom_advice(self, user_text):
        text = user_text.lower()
        for symptom, advice in self.symptom_dict.items():
            if symptom.lower() in text:
                return advice
        return None

    def reply(self, user_input):
        symptom_advice = self.get_symptom_advice(user_input)
        if symptom_advice:
            return f"🩺 Symptom Advice:\n{symptom_advice}"
        else:
            faq_answer = self.get_faq(user_input)
            return f"📋 FAQ Answer:\n{faq_answer}"
