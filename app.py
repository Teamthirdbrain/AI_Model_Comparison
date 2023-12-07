import streamlit as st
from difflib import SequenceMatcher
from transformers import pipeline, GPT2Tokenizer, BertTokenizer, RobertaTokenizer

# AI models and their respective names
models = {
    "ChatGPT": "microsoft/DialoGPT-medium",
    "BERT": "bert-base-uncased",
    "OpenAI GPT": "gpt2",
    "RoBERTa": "roberta-base",
}

# Load the models and their tokenizers
tokenizer_models = {
    "ChatGPT": GPT2Tokenizer.from_pretrained(models["ChatGPT"]),
    "BERT": BertTokenizer.from_pretrained(models["BERT"]),
    "OpenAI GPT": GPT2Tokenizer.from_pretrained(models["OpenAI GPT"]),
    "RoBERTa": RobertaTokenizer.from_pretrained(models["RoBERTa"])
}

generation_models = {
    "ChatGPT": pipeline("text-generation", model=models["ChatGPT"], tokenizer=tokenizer_models["ChatGPT"]),
    "BERT": pipeline("text-generation", model=models["BERT"], tokenizer=tokenizer_models["BERT"]),
    "OpenAI GPT": pipeline("text-generation", model=models["OpenAI GPT"], tokenizer=tokenizer_models["OpenAI GPT"]),
    "RoBERTa": pipeline("text-generation", model=models["RoBERTa"], tokenizer=tokenizer_models["RoBERTa"])
}

# Streamlit app
st.title("AI Model Comparison")

# Input query
benchmark_query = st.text_input("Enter your benchmark query or data:")

if benchmark_query:
    # Generate outputs using each AI model
    outputs = {}
    similarities = {}

    for model_name, model in generation_models.items():
        output = model(benchmark_query, max_length=100, num_return_sequences=1)
        output_text = output[0]["generated_text"].strip()
        similarity = SequenceMatcher(None, benchmark_query, output_text).ratio()

        outputs[model_name] = output_text
        similarities[model_name] = similarity

    # Display the results
    st.subheader("Comparison Results:")
    for model_name, output_text in outputs.items():
        st.markdown(f"**{model_name}**")
        st.info(output_text)
        st.write("---")

    st.subheader("Similarity Scores:")
    for model_name, similarity in similarities.items():
        st.markdown(f"**{model_name}**: {similarity}")

    st.subheader("Recommendation:")
    max_similarity = max(similarities.values())
    recommended_models = [model_name for model_name, similarity in similarities.items() if similarity == max_similarity]

    if len(recommended_models) == 1:
        st.success(f"{recommended_models[0]} provides the most relevant answer.")
        st.success(f"Recommendation: Use {recommended_models[0]} for this task.")
    else:
        st.success("Multiple models provide equally relevant answers.")
        st.success("Recommendation: You can choose any of the recommended models for this task.")
