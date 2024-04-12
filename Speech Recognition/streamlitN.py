#importing libraries 
import streamlit as st
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from streamlit_mic_recorder import mic_recorder
import numpy as np
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from langdetect import detect # package for detection of languages (Hindi + Other languages)
from pydub import AudioSegment

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"

@st.cache_resource
def load_model():
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    return model

# set page wide 
# st.set_page_config(layout="wide")

@st.cache_resource
def load_processor():
    processor = AutoProcessor.from_pretrained(model_id)
    return processor

# torch.save(load_model(), './pre_trained_model/preTrainedModel.pth')
# torch.save(load_processor(), './pre_trained_model/preTrainedProcessor.pth')
model = torch.load('./pre_trained_model/preTrainedModel.pth')
processor = torch.load('./pre_trained_model/preTrainedProcessor.pth')

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=60,
    batch_size=8,
    return_timestamps=False,
    torch_dtype=torch_dtype,
    device=device,
)

# Maintain a list to store the last 5 detected results
@st.cache_resource
def last_results() :
    last_5_results = []
    return last_5_results

detected_language = None
target_language = None
detected_text = None

last_5_results = last_results()

@st.cache_resource
def last_results_video() :
    last_5_results_video = []
    return last_5_results_video

last_5_results_video = last_results_video()

languages = {'gu' : 'gujarati',
             'en' : 'english',
             'hi' : 'hindi',
             'bn' : 'bengali',
             'mr' : 'marathi',
             'ta' : 'tamil'}

#Function for detection of language 
def hindi_to_language(text, detected_language, target_language):
    if detected_language == "hi" and target_language == "Gujarati": #if detected language is hindi and target language is Gujarati
        return transliterate(text, sanscript.DEVANAGARI, sanscript.GUJARATI)
    elif detected_language == "hi" and target_language != "Gujarati": #if detected language is hindi and target language is not Gujarati
        return text  # Keep the text as Hindi
    else:
        # Translate to target language if it's Other language
        return text

# Function to extract audio from video
def extract_audio(video_file):
    audio_file = "temp_audio.wav"
    audio = AudioSegment.from_file(video_file, format="mp4")
    audio.export(audio_file, format="wav")
    return audio_file

def main():
    
    #st.title("Language Detection from Speech")
    st.image('INTELLIVOICE.png',width=200)
    st.write("")
    
    tab1, tab2 = st.tabs(['Audio','Video'])
    global detected_language
    global detected_text
    global target_language
    
    with st.sidebar:
            target_language = st.selectbox(
                "Select Target Language:",
                ("hindi", 'english', 'gujarati', 'bengali', 'marathi', 'tamil'),  # add here other language
            )
            
            # Display the last 5 detected results in the sidebar
            st.markdown("**Last 5 Detected Results**")
            def update_last_5_result() : 
                for result in last_5_results:
                    st.markdown(f"**Detected Language:** {result['detected_language']}, **Text:** {result['text']}")
            st.markdown("---")
            
            
            # def refresh_last_results() :
            #     update_last_5_result()
            # refresh_last_results()
            print(f'last 5 update : {last_5_results}')
            
    with tab1:
        audio = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop recording",
            just_once=True,
            use_container_width=False,
            callback=None,
            args=(),
            kwargs={},
            key=None,
        )

        st.write("OR")
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])


        with st.spinner("Please Wait..."):
            if audio is not None:
                result = pipe(audio["bytes"], generate_kwargs={'language' : target_language})
                detected_text = result["text"]
                detected_language = detect(detected_text)  # Language detection using langdetect
                print(f'DL {detected_language} DT {detected_text}')
                
                
                
            else:
                # File upload widget
                if uploaded_file is not None:
                    bytes_data = uploaded_file.getvalue()
                    result = pipe(bytes_data)
                    detected_text = result["text"]
                    detected_language = detect(detected_text)  # Language detection using langdetect
                    translated_text = hindi_to_language(detected_text, detected_language, target_language)
                
    with tab2:
        uploaded_video = st.file_uploader("Upload a video file", type=["mp4"])
        with st.spinner("Please Wait..."):
            
            if uploaded_video is not None: 
                    try:
                        # Extract audio from the uploaded video file
                        video_file = extract_audio(uploaded_video)
                        print(video_file)
                        with open(video_file, "rb") as f:
                            bytes_data = f.read()
                        # bytes_data=video_file.getvalue()
                        result=pipe(bytes_data)
                        detected_text = result["text"]
                        detected_language = detect(detected_text)  # Language detection using langdetect
                    except Exception as e:
                        print(f"Error: {e}")
                        
    if detected_language is not None:        
                if target_language == languages.get(detected_language) : 
                        translated_text = hindi_to_language(detected_text, detected_language, target_language)
                        st.info(f"Detected Language: {detected_language}")
                        st.info(f"{target_language} Text: {detected_text}")
                        # output only if input is in selected target language
                        if target_language == 'gujarati' :
                            st.write(f"{target_language} Transliteration:", translated_text)
                            
                        # Add the detected result to the last_5_results list
                        print(len(last_5_results))
                        if len(last_5_results) >= 5:
                            last_5_results.pop(0)
                        last_5_results.append({'detected_language': detected_language, 'text': detected_text})
                        print(f'last 5 : {last_5_results}')
                        with st.sidebar :
                            update_last_5_result()
                else :
                    st.warning(f"Please speak in {target_language}!!", icon="⚠️")
                    st.info(f"Detected Language: {detected_language}")
                    st.info(f"Text: {detected_text}")
                audio = None
                uploaded_file = None 
                uploaded_video = None
if __name__ == "__main__":
    main()
