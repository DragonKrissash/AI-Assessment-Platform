"""
AI Assessment Interview Platform
==================================
A Streamlit-based technical interview simulator powered by:
- Groq (LLaMA 3.3 70B) for intelligent question generation & evaluation
- Groq Whisper for speech-to-text transcription
- ElevenLabs for text-to-speech audio responses
- LangChain for conversation history management
- LangSmith for tracing/observability

Flow:
  1. User selects a domain
  2. LLM asks MAX_QUESTIONS technical questions one at a time
  3. User can answer by TYPING or by SPEAKING (press Enter to record, press Enter to stop)
  4. After all answers, LLM generates a final evaluation report
"""

import os
import io
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from elevenlabs.client import ElevenLabs
from groq import Groq
from streamlit_mic_recorder import mic_recorder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

REQUIRED_ENV_VARS = [
    "GROQ_API_KEY",
    "ELEVENLABS_API_KEY",
    "LANGCHAIN_PROJECT",
    "LANGCHAIN_API_KEY",
]
missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing:
    st.error(f"Missing required environment variables: {', '.join(missing)}")
    st.stop()

os.environ["GROQ_API_KEY"]         = os.getenv("GROQ_API_KEY")
os.environ["ELEVENLABS_API_KEY"]   = os.getenv("ELEVENLABS_API_KEY")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

MAX_QUESTIONS     = 5
INTERVIEW_DOMAINS = ["AI/ML", "FrontEnd", "BackEnd", "DevOps"]
ELEVENLABS_VOICE  = "JBFqnCBsd6RMkjVDRZzb"
ELEVENLABS_MODEL  = "eleven_multilingual_v2"
GROQ_MODEL        = "llama-3.3-70b-versatile"
WHISPER_MODEL     = "whisper-large-v3"

# ---------------------------------------------------------------------------
# Cached client initialization
# ---------------------------------------------------------------------------

@st.cache_resource
def get_llm() -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL, streaming=False)


@st.cache_resource
def get_audio_client() -> ElevenLabs:
    return ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


@st.cache_resource
def get_groq_client() -> Groq:
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "domain":         None,
        "messages":       None,
        "question_count": 0,
        "interview_done": False,
        "input_mode":     "type",   # "type" | "speak"
        "last_mic_id":    None,     # track processed mic recordings
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def text_to_speech_bytes(text: str) -> bytes | None:
    """Convert text to MP3 audio bytes via ElevenLabs TTS."""
    if not text or not text.strip():
        return None
    try:
        client       = get_audio_client()
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=ELEVENLABS_VOICE,
            model_id=ELEVENLABS_MODEL,
            output_format="mp3_44100_128",
        )
        return b"".join(audio_stream)
    except Exception as e:
        st.warning(f"⚠️ Audio generation failed: {e}")
        return None


def play_audio(audio_bytes: bytes | None) -> None:
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3", autoplay=True)


def speech_to_text(audio_bytes: bytes) -> str | None:
    """
    Transcribe audio bytes to text using Groq Whisper.

    Args:
        audio_bytes: Raw audio bytes from the mic recorder.

    Returns:
        Transcribed text string, or None on failure.
    """
    if not audio_bytes:
        return None
    try:
        client = get_groq_client()
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"  # Groq needs a filename hint

        transcription = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
            response_format="text",
        )
        return transcription.strip() if transcription else None
    except Exception as e:
        st.error(f"❌ Speech transcription failed: {e}")
        return None


# ---------------------------------------------------------------------------
# LLM utilities
# ---------------------------------------------------------------------------

def build_system_prompt(domain: str) -> str:
    return f"""
You are an expert technical interviewer in {domain} with over 10 years of experience.

STRICT RULES — follow these exactly, without exception:
1. Ask EXACTLY ONE question per response. Never ask two questions in the same message.
2. You may ask a MAXIMUM of {MAX_QUESTIONS} technical questions in total during this interview.
   The opening prompt ("Please introduce yourself") does NOT count toward this limit.
3. After the user answers, briefly evaluate their response in 1-2 sentences, then ask the next question.
4. Adapt difficulty: increase slightly if the previous answer was strong; keep similar or easier if weak.
5. Once all {MAX_QUESTIONS} questions have been asked and answered, do NOT ask any further questions.
6. Keep the tone encouraging, professional, and conversational.
7. NEVER use phrases like "Here are a few questions:" — only ever present one question at a time.
"""


def build_conversation_history(system_prompt: str) -> list:
    history = [SystemMessage(content=system_prompt)]
    for msg in st.session_state.messages:
        role    = msg["role"]
        content = msg["content"]
        if role == "user":
            history.append(HumanMessage(content=content))
        else:
            history.append(AIMessage(content=content))
    return history


def invoke_llm(history: list) -> str | None:
    try:
        llm = get_llm()
        return llm.invoke(history).content
    except Exception as e:
        st.error(f"❌ LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Chat UI helpers
# ---------------------------------------------------------------------------

def render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            prefix = "**Interviewer:** " if msg["role"] == "assistant" else ""
            st.markdown(f"{prefix}{msg['content']}")
            if msg.get("audio"):
                st.audio(msg["audio"], format="audio/mp3", autoplay=False)


def save_user_message(user_input: str, audio_bytes: bytes | None = None) -> None:
    st.session_state.messages.append({
        "role":    "user",
        "content": user_input,
        "audio":   audio_bytes,
    })
    with st.chat_message("user"):
        st.markdown(user_input)


def save_assistant_message(text: str, audio_bytes: bytes | None) -> None:
    st.session_state.messages.append({
        "role":    "assistant",
        "content": text,
        "audio":   audio_bytes,
    })
    with st.chat_message("assistant"):
        st.markdown(f"**Interviewer:** {text}")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3", autoplay=False)


# ---------------------------------------------------------------------------
# Interview logic
# ---------------------------------------------------------------------------

def handle_next_question(system_prompt: str) -> None:
    st.session_state.question_count += 1
    q_num = st.session_state.question_count

    history = build_conversation_history(system_prompt)
    history.append(SystemMessage(content=f"""
        This is question {q_num} of {MAX_QUESTIONS}.
        Briefly evaluate the user's last answer (1-2 sentences),
        then ask question {q_num}.
        Ask ONLY this one question. Do NOT ask question {q_num + 1}.
    """))

    with st.spinner("Thinking..."):
        response_text = invoke_llm(history)

    if response_text:
        audio_bytes = text_to_speech_bytes(response_text)
        play_audio(audio_bytes)
        save_assistant_message(response_text, audio_bytes)


def handle_final_evaluation(system_prompt: str) -> None:
    history = build_conversation_history(system_prompt)
    history.append(SystemMessage(content=f"""
        The candidate has answered all {MAX_QUESTIONS} questions. The interview is now over.
        Do NOT ask any more questions.

        Generate a structured final evaluation report containing:
        - Overall score (out of 10)
        - Performance summary (3-4 sentences)
        - Strong areas (bullet points)
        - Weak areas with specific, actionable improvement suggestions (bullet points)
        - Pass / Fail verdict with a brief justification
    """))

    with st.spinner("Generating your evaluation report..."):
        response_text = invoke_llm(history)

    if response_text:
        audio_bytes = text_to_speech_bytes(response_text)
        play_audio(audio_bytes)
        save_assistant_message(response_text, audio_bytes)
        st.session_state.interview_done = True
        st.rerun()


def process_user_answer(user_input: str) -> None:
    """Central handler: save the answer and route to next question or evaluation."""
    if not user_input or not user_input.strip():
        return

    domain        = st.session_state.domain
    system_prompt = build_system_prompt(domain)

    save_user_message(user_input.strip())

    if st.session_state.question_count < MAX_QUESTIONS:
        handle_next_question(system_prompt)
    else:
        handle_final_evaluation(system_prompt)


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def render_domain_selection() -> None:
    st.subheader("Welcome 👋")
    st.write("Select a technical domain below, then press **Start Interview** to begin.")

    domain = st.pills("Interview domain:", INTERVIEW_DOMAINS)

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Start Interview", type="primary"):
            if domain:
                st.session_state.domain = domain
                st.rerun()
            else:
                st.warning("Please select a domain first.")


def render_input_mode_toggle() -> None:
    """Render the Type / Speak toggle above the input area."""
    st.write("")  # spacing
    col_label, col_toggle = st.columns([2, 3])
    with col_label:
        st.markdown("**Answer mode:**")
    with col_toggle:
        mode = st.radio(
            label="Answer mode",
            options=["⌨️ Type", "🎙️ Speak"],
            horizontal=True,
            label_visibility="collapsed",
            key="mode_radio",
        )
        st.session_state.input_mode = "type" if mode == "⌨️ Type" else "speak"


def render_voice_input() -> None:
    """
    Render the mic recorder widget.

    mic_recorder returns a dict with keys:
      - 'bytes': raw audio bytes (WAV)
      - 'id':    unique int incremented each recording (use to detect new recordings)
    When the user clicks once → starts recording.
    When the user clicks again → stops and returns audio.
    """
    st.info(
        "🎙️ Click **Start recording** to begin speaking, then click **Stop recording** when done. "
        "Your speech will be transcribed and sent automatically.",
        icon="ℹ️",
    )

    mic_key = f"mic_recorder_{st.session_state.question_count}"

    audio = mic_recorder(
        start_prompt="▶ Start recording",
        stop_prompt="⏹ Stop recording",
        just_once=True,          # widget resets after each capture
        use_container_width=True,
        key=mic_key,
    )

    if audio and audio.get("bytes"):
        recording_id = audio.get("id")
        # Guard against reprocessing the same recording on reruns
        if recording_id != st.session_state.last_mic_id:
            st.session_state.last_mic_id = recording_id
            with st.spinner("🔄 Transcribing your answer..."):
                transcript = speech_to_text(audio["bytes"])

            if transcript:
                st.success(f"📝 Transcribed: *{transcript}*")
                process_user_answer(transcript)
                st.rerun()
            else:
                st.warning("⚠️ Could not transcribe audio. Please try again or switch to typing.")


def render_interview() -> None:
    domain        = st.session_state.domain
    system_prompt = build_system_prompt(domain)

    st.caption(
        f"🎯 Domain: **{domain}** | "
        f"Questions answered: **{st.session_state.question_count} / {MAX_QUESTIONS}**"
    )

    # --- First run: initialise opening message ---
    if st.session_state.messages is None:
        opening     = "Please introduce yourself in brief."
        audio_bytes = text_to_speech_bytes(opening)
        play_audio(audio_bytes)
        st.session_state.messages = [{
            "role":    "assistant",
            "content": opening,
            "audio":   audio_bytes,
        }]

    # --- Completed interview screen ---
    if st.session_state.interview_done:
        render_chat_history()
        st.success("✅ Interview complete! Review your evaluation above.")
        if st.button("🔄 Start a New Interview"):
            for key in ["domain", "messages", "question_count", "interview_done",
                        "input_mode", "last_mic_id"]:
                del st.session_state[key]
            st.rerun()
        st.stop()

    # --- Active interview ---
    render_chat_history()

    # Input mode selector
    render_input_mode_toggle()

    # --- Text input mode ---
    if st.session_state.input_mode == "type":
        user_input = st.chat_input("Type your answer here...")
        if user_input:
            process_user_answer(user_input)

    # --- Voice input mode ---
    else:
        render_voice_input()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="AI Assessment Platform",
        page_icon="🎙️",
        layout="centered",
    )
    st.title("🎙️ AI Assessment Platform")

    init_session_state()

    if st.session_state.domain is None:
        render_domain_selection()
    else:
        render_interview()


if __name__ == "__main__":
    main()