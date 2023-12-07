import streamlit as st
from difflib import SequenceMatcher
from transformers import pipeline, GPT2Tokenizer

# AI models and their respective names
models = {
    "gpt2": "GPT-2",
    "gpt2-medium": "GPT-2 Medium",
    "gpt2-large": "GPT-2 Large",
    # Add more models here as per your requirements
}

# Load the models and their tokenizers
tokenizer_models = {model_name: GPT2Tokenizer.from_pretrained(model_name) for model_name in models}
generation_models = {model_name: pipeline("text-generation", model=model_name, tokenizer=tokenizer_models[model_name]) for model_name in models}


# Streamlit app
st.title("ᴀɪ ᴍᴏᴅᴇʟ ᴄᴏᴍᴘᴀʀɪꜱᴏɴ")

# Model selection
selected_models = st.multiselect("ꜱᴇʟᴇᴄᴛ ᴀɪ ᴍᴏᴅᴇʟꜱ ᴛᴏ ᴄᴏᴍᴘᴀʀᴇ:", list(models.keys()))

# Input query
benchmark_query = st.text_input("ᴇɴᴛᴇʀ ʏᴏᴜʀ ʙᴇɴᴄʜᴍᴀʀᴋ Qᴜᴇʀʏ ᴏʀ ᴅᴀᴛᴀ:")

if benchmark_query and len(selected_models) == 2:
    # Generate outputs using the selected models
    output_a = generation_models[selected_models[0]](benchmark_query, max_length=100, num_return_sequences=1)
    output_b = generation_models[selected_models[1]](benchmark_query, max_length=100, num_return_sequences=1)

    # Extract the generated text from the output
    output_a_text = output_a[0]["generated_text"].strip()
    output_b_text = output_b[0]["generated_text"].strip()

    # Compare and evaluate the outputs
    similarity_a = SequenceMatcher(None, benchmark_query, output_a_text).ratio()
    similarity_b = SequenceMatcher(None, benchmark_query, output_b_text).ratio()

    # Display the results
    st.subheader("ᴄᴏᴍᴘᴀʀɪꜱᴏɴ ʀᴇꜱᴜʟᴛꜱ:")

    st.markdown(
        f"<div style='display:flex; gap:20px;'>"
        f"<div style='flex:1; background-color:#2E1813; padding:20px; border-radius:10px;'>"
        f"<h3>{models[selected_models[0]]}</h3>"
        f"<p>{output_a_text}</p>"
        f"</div>"
        f"<div style='flex:1; background-color:#2E1813; padding:20px; border-radius:10px;'>"
        f"<h3>{models[selected_models[1]]}</h3>"
        f"<p>{output_b_text}</p>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.subheader("ꜱɪᴍɪʟᴀʀɪᴛʏ ꜱᴄᴏʀᴇꜱ:")

    st.markdown(
        f"<div style='display:flex; gap:20px;'>"
        f"<div style='background-color:#2E1813; padding:20px; border-radius:10px;'>"
        f"<p>Similarity Score ({models[selected_models[0]]}): {similarity_a}</p>"
        f"</div>"
        f"<div style='background-color:#2E1813; padding:20px; border-radius:10px;'>"
        f"<p>Similarity Score ({models[selected_models[1]]}): {similarity_b}</p>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.subheader("ʀᴇᴄᴏᴍᴍᴇɴᴅᴀᴛɪᴏɴ:")
    if similarity_a > similarity_b:
        st.success(f"{models[selected_models[0]]}ᴘʀᴏᴠɪᴅᴇꜱ ᴀ ᴍᴏʀᴇ ʀᴇʟᴇᴠᴀɴᴛ ᴀɴꜱᴡᴇʀ.")
        st.success(f"Recommendation: Use {models[selected_models[0]]} for this task.")
    elif similarity_a < similarity_b:
        st.success(f"{models[selected_models[1]]}ᴘʀᴏᴠɪᴅᴇꜱ ᴀ ᴍᴏʀᴇ ʀᴇʟᴇᴠᴀɴᴛ ᴀɴꜱᴡᴇʀ.")
        st.success(f"Recommendation: Use {models[selected_models[1]]} for this task.")
    else:
        st.success("​🇧​​🇴​​🇹​​🇭​ ​🇦​​🇮​ ​🇲​​🇴​​🇩​​🇪​​🇱​​🇸​ ​🇵​​🇷​​🇴​​🇻​​🇮​​🇩​​🇪​ ​🇸​​🇮​​🇲​​🇮​​🇱​​🇦​​🇷​ ​🇱​​🇪​​🇻​​🇪​​🇱​​🇸​ ​🇴​​🇫​ ​🇷​​🇪​​🇱​​🇪​​🇻​​🇦​​🇳​​🇨​​🇪​.")
        st.success("​🇷​​🇪​​🇨​​🇴​​🇲​​🇲​​🇪​​🇳​​🇩​​🇦​​🇹​​🇮​​🇴​​🇳​⦂ ​🇾​​🇴​​🇺​ ​🇨​​🇦​​🇳​ ​🇨​​🇭​​🇴​​🇴​​🇸​​🇪​ ​🇪​​🇮​​🇹​​🇭​​🇪​​🇷​ ​🇲​​🇴​​🇩​​🇪​​🇱​ ​🇫​​🇴​​🇷​ ​🇹​​🇭​​🇮​​🇸​ ​🇹​​🇦​​🇸​​🇰​.")
elif benchmark_query or len(selected_models) == 2:
    st.warning("🇵​​🇱​​🇪​​🇦​​🇸​​🇪​ ​🇪​​🇳​​🇹​​🇪​​🇷​ ​🇦​ ​🇧​​🇪​​🇳​​🇨​​🇭​​🇲​​🇦​​🇷​​🇰​ ​🇶​​🇺​​🇪​​🇷​​🇾​ ​🇦​​🇳​​🇩​ ​🇸​​🇪​​🇱​​🇪​​🇨​​🇹​ ​🇪​​🇽​​🇦​​🇨​​🇹​​🇱​​🇾​ ​🇹​​🇼​​🇴​ ​🇲​​🇴​​🇩​​🇪​​🇱​​🇸​.")
