from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

import streamlit as st

from langchain.docstore.document import Document
import whisper
import pyaudio
import wave
import openai

import time
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
from langchain.document_loaders import UnstructuredFileLoader

# GUI Display

st.set_page_config(page_title="KudasAI",page_icon="fox_face",layout="wide",initial_sidebar_state="expanded",menu_items={'Get help':'mailto:akshay@kudasai.io'})
col1, col2, col3 = st.columns(3)

with col2:
    st.header(":fox_face: KudasAI :tm:")
    st.subheader(":sparkles: *beep-boop-beep* :robot_face:")

# get environment path variable OPENAI_API_KEY
# openai.api_key  = st.sidebar.text_input('Your API Key')
# PINECONE_API_KEY = st.sidebar.text_input('Your Pine Key')
# PINECONE_API_ENV = st.sidebar.text_input('Your Pine env')
OPENAI_API_KEY = st.sidebar.text_input('Your API Key')
openai.api_key = OPENAI_API_KEY
PINECONE_API_KEY = PINECONE_API_KEY
PINECONE_API_ENV = 'us-central1-gcp'

# Initialize Pinecone vectorstore

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "reader"

# Initialize Whisper API Transcriber
model = whisper.load_model("base")
status = st.markdown("**Kai Engine Loaded**")

# upload audio file:
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

# RECORD AUDIO TO TEMP_FILE:
temp_audio_file_path = "./output.wav"
CHANNELS = 1
FORMAT = pyaudio.paInt16
RATE = 44109
CHUNK = 1024
THOUGHT_DELAY = int(st.sidebar.select_slider(
    'Recording Time (s)',
    options=['5', '15', '30', '60', '120', '180', '240']))

# Convert to milliseconds:
# def convertMillis(start_ms):
#     seconds = int((start_ms / 1000) % 60)
#     minutes = int((start_ms / (1000 * 60)) % 60)
#     hours = int((start_ms / (1000 * 60 * 60)) % 24)
#     btn_txt = ''
#     if hours > 0:
#         btn_txt += f'{hours:02d}:{minutes:02d}:{seconds:02d}'
#     else:
#         btn_txt += f'{minutes:02d}:{seconds:02d}'
#     return btn_txt

# Define session state
if 'start_point' not in st.session_state:
    st.session_state['start_point'] = 0

# def update_start(start_t):
#     st.session_state['start_point'] = int(start_t/1000)

#Flux functions:
@st.cache_data
def flux1(result):
    #PART1
    # Get a summary from GPT-3
    prompt1=(f"You are a cognitive insights engine.\nYour goal is to create a summary of a meeting transcription in the following format:\n\n1. Mention a header for the topic of discussion\n2. State the high-level topics discussed in the text\n3. Create a precise summary without missing key datapoints\n\nHere is a example of the required output format:\n\n\"Gist: Launching a Getaway Cluster with Research-Based User Experience\n\nSummary: This conversation focused on launching a Getaway Cluster with research-based user experience. They discussed Jira stuff and integration with Visual Studio Code, as well as Cindy's suggestion to list out and call certain key iterations something overall. It was decided that a UX bucket should be included as it is typical for big launches, and the excitement level was assessed to be a three. The only way to make sense of the iterations was to bucket them with something, and the suggestion was accepted. \"\n\ntext input: {result}\n\nYour Response:\n")
    response = openai.Completion.create(model="text-davinci-003",prompt=prompt1,max_tokens=1024,n =1,stop=None,temperature=0.7,)

    # Get the summary from the response
    summary = response.choices[0].text
    return summary

@st.cache_data
def flux2(result):
    #PART2
    # Get insights from GPT-3
    prompt2=(f"You are a cognitive insights engine.\nYour goal is to develop insights of a meeting transcription in the following format:\n\n1. Provide insights to help the reader understand the stakeholders of the discussion, by making the dependencies of the topics clear with subheadings for each\n2. If any actionable points are mentioned, plan a roadmap with consequent steps to initiate the project in bullet points\n\nHere is a example of an output format:\n\n\"Insights: \n\n- Integration with Visual Studio Code: This was mentioned as a potential notable area for the launch\n- Key Iterations: Cindy suggested listing and calling this something overall, and this suggestion was accepted\n- UX Bucket: This was suggested as it is typically included in big launches, and accepted \n- Excitement Level: This was assessed to be a three\n\Further Considerations: \n\n- Continue integration with Visual Studio Code\n- List out and call the key iterations\n- Include a UX bucket as part of the launch \n- Assess the excitement level \n- Bucket the iterations with something\"\n\nThe text input to be processed {result}\n\nYour Response:\n")
    response = openai.Completion.create(model="text-davinci-003",prompt=prompt2,max_tokens=1024,n =1,stop=None,temperature=0.7,)

    # Get the insight from the response
    insight = response.choices[0].text
    return insight

@st.cache_data
def flux3(text_file):
    #PART1
    # Get a summary from GPT-3
    prompt1=(f"You are a cognitive insights engine.\nYour goal is to create a summary of a meeting transcription in the following format:\n\n1. Mention a header for the topic of discussion\n2. State the high-level topics discussed in the text\n3. Create a precise summary without missing key datapoints\n\nHere is a example of the required output format:\n\n\"Gist: Launching a Getaway Cluster with Research-Based User Experience\n\nSummary: This conversation focused on launching a Getaway Cluster with research-based user experience. They discussed Jira stuff and integration with Visual Studio Code, as well as Cindy's suggestion to list out and call certain key iterations something overall. It was decided that a UX bucket should be included as it is typical for big launches, and the excitement level was assessed to be a three. The only way to make sense of the iterations was to bucket them with something, and the suggestion was accepted. \"\n\ntext input: {text_file}\n\nYour Response:\n")
    response = openai.Completion.create(model="text-davinci-003",prompt=prompt1,max_tokens=1024,n =1,stop=None,temperature=0.7,)

    # Get the summary from the response
    summary = response.choices[0].text
    return summary

@st.cache_data
def flux4(text_file):
    #PART2
    # Get insights from GPT-3
    prompt2=(f"You are a cognitive insights engine.\nYour goal is to develop insights of a meeting transcription in the following format:\n\n1. Provide insights to help the reader understand the stakeholders of the discussion, by making the dependencies of the topics clear with subheadings for each\n2. If any actionable points are mentioned, plan a roadmap with consequent steps to initiate the project in bullet points\n\nHere is a example of an output format:\n\n\"Insights: \n\n- Integration with Visual Studio Code: This was mentioned as a potential notable area for the launch\n- Key Iterations: Cindy suggested listing and calling this something overall, and this suggestion was accepted\n- UX Bucket: This was suggested as it is typically included in big launches, and accepted \n- Excitement Level: This was assessed to be a three\n\Further Considerations: \n\n- Continue integration with Visual Studio Code\n- List out and call the key iterations\n- Include a UX bucket as part of the launch \n- Assess the excitement level \n- Bucket the iterations with something\"\n\nThe text input to be processed {text_file}\n\nYour Response:\n")
    response = openai.Completion.create(model="text-davinci-003",prompt=prompt2,max_tokens=1024,n =1,stop=None,temperature=0.7,)

    # Get the insight from the response
    insight = response.choices[0].text
    return insight

# Transcription via OpenAI
@st.cache_data
def trans(temp_audio_file_path):
    st.markdown("Transcribing Audio...")
    text_file = model.transcribe(temp_audio_file_path)["text"]
    with st.expander('Transcription text'):
        st.markdown(text_file)
        file = open('transcript_temp.txt', 'w')
        file.write(text_file)
        file.close()
    return text_file

@st.cache_data
def query_doc(query):
    loader = UnstructuredFileLoader('./transcript_temp.txt')
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")

    docs = docsearch.similarity_search(query, include_metadata=True)
    st.success('Done!')

    result = chain.run(input_documents=docs, question=query)
    return result

# Text splitter via LangChain:
# @st.cache_data
# def text_split(transcript):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=10)
#     splittexts = text_splitter.split_text(transcript)
#     docs = [Document(page_content=t) for t in splittexts[:len(splittexts)]]
#     return docs

# RECORD_AUDIO:
@st.cache_data
def Rec_audio():
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)
        st.markdown("_Go ahead, I'm listening..._")
        frames = []

        for i in range(0, int(RATE / CHUNK * THOUGHT_DELAY)):
            data = stream.read(CHUNK)
            frames.append(data)

        st.markdown("Recording done. Processing audio file")
        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(temp_audio_file_path, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        st.markdown(f"Audio file written at: {temp_audio_file_path}")
        st.audio(temp_audio_file_path, start_time=st.session_state['start_point'])
        
        transcript = trans(temp_audio_file_path=temp_audio_file_path)
        # Text splitter via LangChain:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=10)
        splittexts = text_splitter.split_text(transcript)
        # print(len(splittexts))
        docs = [Document(page_content=t) for t in splittexts[:len(splittexts)]]

        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

        # if statement for short vs long text (chunking):
        if len(splittexts) > 1:
            
            chain = load_summarize_chain(llm, chain_type="refine")
            result = chain.run(docs)
            
            progress_text = "Processing using Flux Capacitor"
            my_bar = st.progress(0, text=progress_text)
            
            for percent_complete in range(100):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1, text=progress_text)
            
            summary1 = flux1(result=result)
            insight2 = flux2(result=result)

            # Print summary & insight
            st.markdown("### Final Summary:")
            st.markdown(f'{summary1} \n\n {insight2}')
            st.success('Done!')

        else:
            progress_text = "Processing using Flux Capacitor"
            my_bar = st.progress(0, text=progress_text)
            
            for percent_complete in range(100):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1, text=progress_text)

            summary3 = flux3(text_file=transcript)
            insight4 = flux4(text_file=transcript)

            # Print summary & insight
            st.markdown("### Final Summary:")
            st.markdown(f'{summary3} \n\n {insight4}')
            st.success('Done!')

if st.sidebar.button("Record Audio"):
    Rec_audio()

    query = st.text_input("Query the transcript")

    if query:
        st.markdown("Mapping the conversation...")
        with st.spinner('Operation in progress. Please wait.'):
            time.sleep(5)
            q_result = query_doc(query=query)
            st.markdown(q_result)
        st.success(st.balloons())


@st.cache_data
def main(): 
    # UPLOAD_FILE:
    st.audio(audio_file, start_time=st.session_state['start_point'])
    file_transcript = trans(temp_audio_file_path=audio_file.name)

    # Text splitter via LangChain:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=10)
    splittexts = text_splitter.split_text(file_transcript)
    docs = [Document(page_content=t) for t in splittexts[:len(splittexts)]]
    
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

    # if statement for short vs long text (chunking):
    if len(splittexts) > 1:

        chain = load_summarize_chain(llm, chain_type="refine",verbose=True)
        result = chain.run(docs)

        progress_text = "Processing using Flux Capacitor"
        my_bar = st.progress(0, text=progress_text)
        
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1, text=progress_text)
        
        summary1 = flux1(result=result)
        insight2 = flux2(result=result)

        # Print summary & insight
        st.markdown("### Final Summary:")
        st.markdown(f'{summary1} \n\n {insight2}')
        st.success('Done!')

    else:
        with st.spinner('Processing using Flux Capacitor'):
            time.sleep(5)
        
        summary3 = flux3(text_file=file_transcript)
        insight4 = flux4(text_file=file_transcript)

        # Print summary & insight
        st.markdown("### Final Summary:")
        st.markdown(f'{summary3} \n\n {insight4}')
        st.success('Done!')

if audio_file is not None:
    st.sidebar.success("Audio File Uploaded")
    app_run = main()

    query = st.text_input("Query the transcript")

    if query:
        st.markdown("Mapping the conversation...")
        with st.spinner('Operation in progress. Please wait.'):
            time.sleep(5)
            q_result = query_doc(query=query)
            st.markdown(q_result)
        st.success(st.balloons())
