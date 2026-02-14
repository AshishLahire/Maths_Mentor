"""
Math Mentor AI - Complete Application
JEE Level Math Assistant with Multi-Agent System
"""

# ================== BASIC SETUP ==================
import os
import tempfile
import streamlit as st
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import shutil

st.write("Tesseract path:", shutil.which("tesseract"))


# Streamlit + Torch watcher issue fix
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"

load_dotenv()

# Tesseract path (Windows)
# try:
#     pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# except:
#     pass

# ================== CUSTOM IMPORTS ==================
from utils.ocr_handler import extract_text_from_image
from utils.audio_handler import transcribe_audio, transcribe_recording

from agents.parser_agent import ParserAgent
from agents.solver_agent import SolverAgent
from agents.verifier_agent import VerifierAgent
from agents.explainer_agent import ExplainerAgent

from rag.retriever import retrieve_context
from memory.simple_memory_handler import MemoryHandler

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Math Mentor AI",
    page_icon="🧮",
    layout="wide"
)

# ================== SESSION STATE ==================
if "memory" not in st.session_state:
    st.session_state.memory = MemoryHandler()

if "input_method" not in st.session_state:
    st.session_state.input_method = "text"

if "question_text" not in st.session_state:
    st.session_state.question_text = ""

if "show_memory" not in st.session_state:
    st.session_state.show_memory = False

if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 0.7

# ================== SIDEBAR ==================
with st.sidebar:
    st.title("Math Mentor AI")

    st.subheader("Input Method")
    method = st.radio(
        "Choose input",
        ["Text", "Image", "Audio"],
        label_visibility="collapsed"
    )
    st.session_state.input_method = method.lower()

    st.divider()

    st.subheader("Confidence Threshold")
    st.session_state.confidence_threshold = st.slider(
        "Accuracy",
        0.0, 1.0, 0.7, 0.05,
        label_visibility="collapsed"
    )

    st.divider()

    st.session_state.show_memory = st.toggle("Show Memory Bank")

    st.divider()
    if os.getenv("GROQ_API_KEY"):
        st.success("🤖 AI Connected")
    else:
        st.error("❌ GROQ_API_KEY missing")

# ================== HEADER ==================
st.markdown("## 🧠 Math Mentor AI (JEE Level)")
st.markdown("Type, upload image, or speak your math problem.")

st.divider()

# ================== INPUT SECTION ==================
st.markdown("### ✍️ Enter Question")

question_text = ""

# -------- TEXT --------
if st.session_state.input_method == "text":
    text = st.text_area(
        "Type your question",
        height=150,
        placeholder="Example: Find the derivative of x^2 + 3x"
    )
    if text:
        question_text = text.strip()
        st.session_state.question_text = question_text

# -------- IMAGE --------
elif st.session_state.input_method == "image":
    img_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    if img_file:
        img = Image.open(img_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Extract Text", use_container_width=True):
            with st.spinner("Extracting text..."):
                text, conf = extract_text_from_image(img)
                st.session_state.question_text = text
                st.success(f"OCR Confidence: {conf:.1%}")

        if st.session_state.question_text:
            edited = st.text_area(
                "Edit extracted text",
                value=st.session_state.question_text,
                height=120
            )
            st.session_state.question_text = edited
            question_text = edited

# -------- AUDIO --------
elif st.session_state.input_method == "audio":
    audio = st.file_uploader(
        "Upload audio",
        type=["wav", "mp3", "m4a", "ogg"]
    )

    if audio:
        st.audio(audio)

        if st.button("🎧 Transcribe Audio", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio.read())
                path = tmp.name

            text, conf = transcribe_audio(path)
            os.unlink(path)

            st.success(f"Confidence: {conf:.1%}")
            edited = st.text_area("Transcribed text", value=text, height=120)
            st.session_state.question_text = edited
            question_text = edited

# ================== SOLVE BUTTON ==================
st.divider()

if st.button("🚀 Solve Math Problem", type="primary", use_container_width=True):

    question_text = st.session_state.question_text

    if not question_text or len(question_text) < 5:
        st.warning("Please enter a valid math question")
        st.stop()

    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY missing in .env")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    try:
        # STEP 1: PARSE
        status.text("🔍 Parsing question...")
        parser = ParserAgent()
        parsed = parser.parse(question_text)
        progress.progress(20)

        # STEP 2: RAG
        status.text("📚 Retrieving context...")
        context = retrieve_context(parsed["problem_text"])
        progress.progress(40)

        # STEP 3: SOLVE
        status.text("🧠 Solving...")
        solver = SolverAgent(use_llm=True)
        solution = solver.solve(parsed, context)
        progress.progress(60)

        # STEP 4: VERIFY
        status.text("✅ Verifying...")
        verifier = VerifierAgent()
        verification = verifier.verify(solution, parsed)
        progress.progress(80)

        # STEP 5: EXPLAIN
        status.text("📖 Explaining...")
        explainer = ExplainerAgent(use_llm=True)
        explanation = explainer.explain(solution, verification)
        progress.progress(100)

        # ================== OUTPUT ==================
        st.success("🎉 Solution Ready!")

        st.markdown("### 🎯 Final Answer")
        st.code(solution.get("final_answer", "N/A"))

        st.markdown("### 📘 Explanation")
        st.markdown(explanation.get("explanation", ""))

        st.markdown("### 📊 Confidence")
        st.write(f"{verification.get('confidence', 0):.2%}")

        # SAVE MEMORY
        st.session_state.memory.store(
            original_input=question_text,
            parsed_problem=parsed,
            solution=solution,
            verification=verification,
            explanation=explanation
        )

    except Exception as e:
        st.error("❌ Something went wrong")
        st.exception(e)

# ================== MEMORY BANK ==================
if st.session_state.show_memory:
    st.divider()
    st.markdown("### 🗂 Memory Bank")

    for mem in st.session_state.memory.get_all(limit=10):
        with st.expander(mem.get("parsed_problem", {}).get("problem_text", "Problem")):
            st.write(mem)

# ================== FOOTER ==================
st.divider()
st.caption("Built with ❤️ | Math Mentor AI | JEE Level")
