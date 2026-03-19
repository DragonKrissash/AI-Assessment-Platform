import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['ELEVENLABS_API_KEY'] = os.getenv('ELEVENLABS_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

audio_lab = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))

# --- Session state initialization ---
if 'question_count' not in st.session_state:
    st.session_state.question_count = 0  # Tracks how many LLM questions have been asked

if 'interview_done' not in st.session_state:
    st.session_state.interview_done = False

MAX_QUESTIONS = 5

st.title('AI Assessment Platform')


# ✅ FIX: Removed text[:10] truncation
def text_to_speech_bytes(text):
    text=text[:10]
    audio_stream = audio_lab.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    return b"".join(audio_stream)


def play_audio(audio_bytes):
    st.audio(audio_bytes, format="audio/mp3", autoplay=True)


def get_history(system_prompt):
    history = [SystemMessage(content=system_prompt)]
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            history.append(HumanMessage(content=msg['content']))
        else:
            history.append(AIMessage(content=msg['content']))
    return history


def refresh_chat():
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            if msg['role'] == 'assistant':
                st.markdown(f"**Interviewer:** {msg['content']}")
            else:
                st.markdown(msg['content'])
            if 'audio' in msg and msg['audio']:
                st.audio(msg['audio'], format="audio/mp3", autoplay=False)


def update_user(user_input, audio_bytes):
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input,
        'audio': audio_bytes
    })
    with st.chat_message("user"):
        st.markdown(user_input)


def update_interviewer(full_text, audio_bytes):
    with st.chat_message("assistant"):
        st.markdown(f"**Interviewer:** {full_text}")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3", autoplay=False)
    st.session_state.messages.append({
        'role': 'assistant',
        'content': full_text,
        'audio': audio_bytes
    })


# --- Domain selection screen ---
if 'domain' not in st.session_state:
    domain = st.pills(
        'Select the domain you want to be interviewed',
        ['AI/ML', 'FrontEnd', 'BackEnd', 'Devops']
    )
    if st.button('Start interview') and domain:
        st.session_state['domain'] = domain
        st.rerun()

# --- Interview screen ---
if 'domain' in st.session_state:

    # ✅ FIX: System prompt strictly enforces question count at the LLM level
    system_prompt = f"""
    You are an expert technical interviewer in {st.session_state['domain']} with over 10 years of experience.

    STRICT RULES — follow these exactly:
    1. Ask EXACTLY ONE question per response. Never ask two questions in the same message.
    2. You are allowed to ask a MAXIMUM of {MAX_QUESTIONS} technical questions in total.
       The opening ("Please introduce yourself") does NOT count toward this limit.
    3. After the user answers, briefly evaluate their response (1-2 sentences), then ask the next question.
    4. If the user's answer was strong, increase difficulty slightly. If weak, keep it similar or easier.
    5. After all {MAX_QUESTIONS} questions have been asked and answered, do NOT ask any more questions.
    6. Keep the tone encouraging and conversational.
    7. NEVER say "Here are a few questions:" — only ever ask one at a time.
    """

    llm = ChatGroq(model='llama-3.3-70b-versatile', streaming=False)

    # Initialize messages with the opening question
    if 'messages' not in st.session_state:
        opening = 'Please introduce yourself in brief.'
        audio_bytes = text_to_speech_bytes(opening)
        play_audio(audio_bytes)
        st.session_state['messages'] = [{
            'role': 'assistant',
            'content': opening,
            'audio': audio_bytes
        }]

    # Show completed interview state
    if st.session_state.interview_done:
        refresh_chat()
        st.info("✅ Interview complete! Review your evaluation above.")
        st.stop()

    refresh_chat()

    user_input = st.chat_input(placeholder='Enter your answer')

    if user_input:
        user_audio = text_to_speech_bytes(user_input)
        play_audio(user_audio)
        update_user(user_input=user_input, audio_bytes=user_audio)

        # ✅ FIX: question_count tracks LLM questions asked (not user answers)
        # Increment BEFORE invoking LLM so count is always accurate
        if st.session_state.question_count < MAX_QUESTIONS:
            st.session_state.question_count += 1

            history = get_history(system_prompt)
            # ✅ FIX: Tell LLM exactly which question number to ask — prevents extras
            history.append(SystemMessage(content=f"""
                This is question {st.session_state.question_count} of {MAX_QUESTIONS}.
                Briefly evaluate the user's last answer (1-2 sentences), then ask question {st.session_state.question_count}.
                Ask ONLY this one question. Do NOT ask question {st.session_state.question_count + 1}.
            """))

            response = llm.invoke(history)
            full_text = response.content
            llm_audio = text_to_speech_bytes(full_text)
            play_audio(llm_audio)
            update_interviewer(full_text=full_text, audio_bytes=llm_audio)

        else:
            # ✅ FIX: Final eval reliably triggers after MAX_QUESTIONS answers
            history = get_history(system_prompt)
            history.append(SystemMessage(content=f"""
                The candidate has now answered all {MAX_QUESTIONS} questions. The interview is over.
                Do NOT ask any more questions.
                Generate the final evaluation report with:
                - Overall score (out of 10)
                - Performance summary
                - Strong areas
                - Weak areas and specific ways to improve
                - Pass / Fail verdict
            """))

            response = llm.invoke(history)
            full_text = response.content
            llm_audio = text_to_speech_bytes(full_text)
            play_audio(llm_audio)
            update_interviewer(full_text=full_text, audio_bytes=llm_audio)

            # ✅ FIX: Set flag then rerun so the done state persists correctly
            st.session_state.interview_done = True
            st.rerun()