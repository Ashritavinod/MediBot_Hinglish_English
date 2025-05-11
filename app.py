import streamlit as st
from chatbot import MediBot
from hinglish_translator import load_translation_dict, translate

st.title("🧠 MediBot – Hinglish + Symptom-Aware Chatbot")
bot = MediBot("C:/Users/ashri\Downloads\MediBot_Chatbot_Streamlit\data\clinic_faqs_translated.csv", "C:/Users/ashri\Downloads\MediBot_Chatbot_Streamlit\data\symptom_advice.csv")
translation_dict = load_translation_dict("C:/Users/ashri\Downloads\MediBot_Chatbot_Streamlit\data\hinglish_to_english.csv")

user_input = st.text_input("💬 Ask something about your health or clinic...")

if user_input:
    translated = translate(user_input, translation_dict)
    response = bot.reply(translated)
    st.markdown(f"**You:** {user_input}")
    st.markdown(f"**MediBot:** {response}")