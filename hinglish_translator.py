import pandas as pd

def load_translation_dict(path):
    df = pd.read_csv(path)
    return dict(zip(df['hinglish'], df['english']))

def translate(text, translation_dict):
    for hing, eng in translation_dict.items():
        if hing in text.lower():
            return eng
    return text