from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI
import requests
import os
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

client = OpenAI(api_key=OPENAI_API_KEY)

#img2text

def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]["generated_text"]

    print(text)
    return text

img2text("pingu.png")

#llm

def generate_story(scenario):
    template = """
    You are a story teller;
    You can generate a short story based on a simple narrative, the story should be no more than 20 words;

    CONTEXT: {}
    STORY:
    """.format(scenario)

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",  # Use GPT-3.5 Turbo model
        prompt=template,  # Input prompt for text generation
        max_tokens=100,  # Maximum number of tokens to generate
        n=1,  # Number of completions to generate
        temperature=0.8,  # Controls the randomness of the generated text
        frequency_penalty=0.5,
        stop=None  # Reduce repetition
)

    generated_text = response.choices[0].text.strip()

    print(generated_text)
    return generated_text

#text to speech

def text2speech(story):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": story
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)

scenario = img2text("pingu.png")
story = generate_story(scenario)
text2speech(story)

#ui

def main():

    st.set_page_config(page_title="img 2 audio story", page_icon="🤖")

    st.header("Turn an Image into an Audio Story")
    uploaded_file = st.file_uploader("Choose an image...", type="png")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
                file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image. ',
                use_column_width=True)
        scenario = img2text(uploaded_file.name)
        scenario_removed_story = generate_story(scenario.replace("CONTEXT:", "").strip())
        text2speech(scenario_removed_story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(scenario_removed_story)

        st.audio("audio.flac")

if __name__ == "__main__":
    main()