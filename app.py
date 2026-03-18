import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

st.title('AI Assessment Platform')


if 'domain' not in st.session_state:
    domain=st.pills(
        'Select the domain you want to be interviewed',
        ['AI/ML','FrontEnd','BackEnd','Devops']
    )
    if st.button('Start interview'):
        st.session_state['domain']=domain
        st.rerun()



if 'question_count' not in st.session_state:
    st.session_state.question_count=0

max_que_count=5

if 'domain' in st.session_state:
    system_prompt = f"""
    You are an expert in {st.session_state['domain']} with over 10 years of experience.

    Rules:
    - Ask ONLY ONE question at a time
    - After user answers, evaluate it briefly 
    - Then ask the NEXT question based on answer quality
    - Slightly increase the question difficulty if answer is good and decrease if answer is incorrect or not completely correct
    - Do NOT ask multiple questions at once
    - Keep the language encouraging
    - Keep interview conversational
    """

def get_history():
    history=[SystemMessage(content=system_prompt)]
    for msg in st.session_state.messages:
        if msg['role']=='user':
            history.append(HumanMessage(content=msg['content']))
        else:
            history.append(AIMessage(content=msg['content']))
    return history

llm=ChatGroq(model='llama-3.3-70b-versatile',streaming=True)

if ('messages' not in st.session_state) and ('domain' in st.session_state):
    st.session_state['messages']=[
        {
            'role':'assistant',
            'content':'Please introduce yourself in brief'
        }
    ]
    st.session_state.question_count+=1

if 'messages' in st.session_state:
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            if msg['role']=='assistant':
                st.markdown(f"**Interviewer:** {msg['content']}")
            else:
                st.markdown(msg['content'])
    if st.session_state.question_count<max_que_count:
        user_input=st.chat_input(placeholder='Enter your answer')

    if st.session_state.question_count==max_que_count:
        history=get_history()
        history.append(SystemMessage(content="""
            Generate the final evaluation of the candidate.
            Give detail summary and score, explaining his weaknesses and strengths.
            Mention the weak areas, strong areas and ways to improve.
            Tell whether pass/fail.
            Anything more you feel important.
        """))
        placeholder=st.empty()
        full_text=''
        for chunk in llm.stream(history):
            full_text+=chunk.content or ''
            placeholder.markdown(f"**Interviewer:** {full_text}")
        st.session_state.messages.append({
            'role':'assistant',
            'content':full_text
        })
        st.stop()
    elif user_input and st.session_state.question_count<10:
        st.session_state.messages.append({
            'role':'user',
            'content':user_input
        })
        history=get_history()
        st.session_state.question_count+=1
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message('assistant'):
            placeholder=st.empty()
            full_text=''

            for chunk in llm.stream(history):
                full_text+=chunk.content or ''
                placeholder.markdown(f"**Interviewer:** {full_text}")
        st.session_state.messages.append({
            'role':'assistant',
            'content':full_text
        })