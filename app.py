import streamlit as st
from chatbot import MediBot
from hinglish_translator import load_translation_dict, translate

st.title("MediBot-Hinglish/English")
bot = MediBot("clinic_faqs_translated.csv","symptom_advice.csv")
translation_dict = load_translation_dict("hinglish_to_english.csv")

user_input = st.text_input("💬 Ask something about your health or clinic...")

if user_input:
    translated = translate(user_input, translation_dict)
    response = bot.reply(translated)
    st.markdown(f"**You:** {user_input}")
    st.markdown(f"**MediBot:** {response}")
