import streamlit as st
import moviepy.editor as mp
from openai import OpenAI
import os
import whisper
from gtts import gTTS
import time
from textwrap import wrap
from pinecone import Pinecone, ServerlessSpec
from summa.summarizer import summarize
import requests
from bs4 import BeautifulSoup 

pc = Pinecone(api_key=st.secrets['DB_TOKEN'])

index_name = "truthspace-index"
client = OpenAI(api_key=st.secrets['AI_TOKEN'])
if index_name not in pc.list_indexes().names():
  pc.create_index(
      name=index_name,
      dimension=768,
      metric="cosine",
      spec=ServerlessSpec(
          cloud='aws', 
          region='us-east-1'
      ) 
  )

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/gtr-t5-base")

def extract_text(soup):
    paragraphs = soup.find_all('p')
    article_text = ' '.join([p.get_text() for p in paragraphs])
    return article_text

def get_similar(claim: str,name: str,count:int = 3):
    index = pc.Index(name)
    embeddings = model.encode(claim)
    output = index.query(vector=embeddings.flatten().tolist(),top_k=count,include_values=True,include_metadata=True)
    return output["matches"]


wmodel = whisper.load_model("base")

def create_audio(text, filename):
    print(text)
    tts = gTTS(text=text, lang='en')
    tts.save(filename)

def factcheck(claim: str,transcript=None):
    matches = get_similar(claim,index_name)
    article = ""
    for i in matches:
        if i['score'] > 0.6:
            url = i['metadata']['url']
            r = requests.get(url)
            content = extract_text(BeautifulSoup(r.content,"html.parser"))
            article += "\n"+content+"\nArticle from: "+url+"\n"
    tran = ""
    if transcript is not None:
        print("made prompt have transcript")
        tran = "\n\nThere was a video attached this is the transcript please mention this as a source and talk about it: \n"+transcript
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "(You can use markdown) Is this claim true or false: "+claim+", Based on th(is/ese) article:\n"+article + tran +"\n Please try to go into more depth and use markdown to make it look prettier and NO MATTER WHAT CITE YOUR SOURCES PROPERLY AT THE END OF YOUR RESPONSE if there is also a transcription of a video MENTION IT also if one of the articles or the article is not related then don't mention it and please try to focus on the claim and not on thecontent of the articles as if i ask you if something happend i can hear about rumors about it but i ultimatley want the claim to be answered true or false"

            }
            ]
        }
    ],
    temperature=0.1,
    max_tokens=1024,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stream = True
    )  
    return response

def makevideo(input: str,transcript=None):
    matches = get_similar(input,index_name)
    article = ""
    for i in matches:
        if i['score'] > 0.6:
            url = i['metadata']['url']
            r = requests.get(url)
            content = extract_text(BeautifulSoup(r.content,"html.parser"))
            article += "\n"+content+"\nArticle from: "+url+"\n"

    tran = ""
    if transcript is not None:
        tran = "\n\nThere was a video attached this is the transcript please mention this as a source and talk about it: \n"+transcript

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "(you can't use markdown) Is this claim true or false: "+input+", Based on th(is/ese) article:\n"+article + tran +"\n This is the script for a video that I am going to generate please seperate each section with [sect] have each section be like 1-2 scentences and if you do cite only put it at the end as these are being compiled into one video also if the article or one of the articles is not related the do not mention it and please try to focus on the claim and not on thecontent of the articles as if i ask you if something happend i can hear about rumors about it but i ultimatley want the claim to be answered true or false"
            }
            ]
        }
    ],
    temperature=1,
    max_tokens=1024,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    sections = response.choices[0].message.content.split("[sect]")
    sections.pop(0)
    clips = []
    for section in sections:
        print(section)
        print("Started clip generation for section:", section)
        audio_filename = "ttstmp.mp3"
        create_audio(section, audio_filename)


        audio = mp.AudioFileClip(audio_filename)
        print("Finished TTS for section:", section)


        wrapped_text = "\n".join(wrap(section, width=40))
        text_clip = mp.TextClip(wrapped_text, fontsize=30, color="white", size=(1280, 720))
        text_clip = text_clip.set_duration(audio.duration + 0.5)
        text_clip = text_clip.set_audio(audio)
        print("Finished creating clip for section:", section)

        clips.append(text_clip)


        os.remove(audio_filename)
        print("Removed temporary audio file for section:", section)

    outvid = mp.concatenate_videoclips(clips, method="compose")
    outvid.write_videofile("ai.mp4", fps=24)
def upload(link: str,name: str):
    index = pc.Index(name)
    r = requests.get(link)
    content = extract_text(BeautifulSoup(r.content,"html.parser"))
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "Please summarize the page from the html provided:\n"+content+"\nthe link to the page is: "+link+"\nplease only give me the summary and nothing else as your ouput is going into a program"
            }
            ]
        }
    ],
    temperature=0,
    max_tokens=4096,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    summary = response.choices[0].message.content
    embeddings = model.encode(summary)
    index.upsert(vectors=[{"id":summary[:512],"values":embeddings.flatten().tolist(),"metadata":{"url":link}}])


st.title("Fact Hawk\n\n")
ask, submit = st.tabs(['Ask!','Submit'])
with ask:
    input = st.text_area("Claim",height=20)
    uploaded_file = st.file_uploader("Have a video?")
    make_video = st.checkbox("Generate Video?")
    if st.button("Ask!",help="Ask the hawk"):
        transcript = None
        if uploaded_file is not None:
            name = "tmp."+uploaded_file.name.split(".")[-1]
            outfile = open(name,"xb")
            outfile.write(uploaded_file.getvalue())
            clip = mp.VideoFileClip(name)
            clip.audio.write_audiofile("tmp.mp3")
            os.remove(name)
            transcript = wmodel.transcribe("tmp.mp3")["text"]
            print("transcribed video: \n"+transcript)
            os.remove("tmp.mp3")
        st.write_stream(factcheck(input,transcript))
        if make_video:
            with st.spinner('Generating Video'):
                makevideo(input,transcript)
                st.video("ai.mp4")

with submit:
    input = st.text_input("Url")
    if st.button("Submit!",help="Help the hawk"):
        with st.spinner('Uploading...'):
            upload(input,index_name)
        st.success("You have submitted a url thank you", icon="✅")
