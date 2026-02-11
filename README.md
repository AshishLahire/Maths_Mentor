# Math Mentor AI - JEE Level Assistant

An AI-powered math tutoring system that solves problems through multiple input methods (text, image, audio) using a multi-agent architecture.

local working link : http://localhost:8501/

## ✨ Features

- **Multi-Input Support**: Text, Image (OCR), Audio (Speech-to-Text)
- **Multi-Agent System**: Parser, Solver, Verifier, Explainer agents
- **RAG Pipeline**: Retrieval-Augmented Generation for contextual answers
- **Live Audio Recording**: Record and transcribe in real-time
- **Memory System**: Stores and recalls past problems
- **Human-in-the-Loop**: User feedback integration
- **JEE Level Explanations**: Detailed step-by-step solutions

## 🚀 Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/AshishLahire/Maths_Mentor.git
cd math-mentor-ai
Install dependencies:

bash
pip install -r requirements.txt
Set up environment variables:

bash
cp .env.example .env
# Add your Groq API key to .env
Run the application:

bash
streamlit run app.py
🔧 Configuration
Environment Variables (.env)
env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
OPENAI_API_KEY=optional_openai_key
OCR Setup
Install Tesseract OCR from here

Set path in code if needed

Audio Setup
bash
# Install audio dependencies
pip install SpeechRecognition pydub streamlit-mic-recorder

# Windows specific
pip install pipwin
pipwin install pyaudio
🏗️ Architecture
text
┌─────────────────────────────────────────────┐
│                 User Interface              │
│  (Text/Image/Audio Input via Streamlit)    │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│              Input Processing               │
│  • OCR Handler (Image → Text)              │
│  • Audio Handler (Audio → Text)            │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│             Multi-Agent System              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│  │ Parser  │ │ Solver  │ │Verifier │      │
│  │ Agent   │ │ Agent   │ │ Agent   │      │
│  └─────────┘ └─────────┘ └─────────┘      │
│         │           │           │          │
│         └─────────┬─┴─────────┬─┘          │
│                   ▼           ▼            │
│             ┌─────────┐ ┌─────────┐       │
│             │  RAG    │ │Explainer│       │
│             │ System  │ │ Agent   │       │
│             └─────────┘ └─────────┘       │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│              Output Display                 │
│  • Step-by-step Solution                   │
│  • Confidence Score                        │
│  • JEE Level Explanation                   │
│  • Memory Storage                          │
└─────────────────────────────────────────────┘
📦 Project Structure
text
math-mentor-ai/
├── app.py                      # Main Streamlit application
├── .env                        # Environment variables
├── .gitignore                  # Git ignore file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── agents/                     # AI Agents
│   ├── parser_agent.py        # Parses questions
│   ├── solver_agent.py        # Solves problems
│   ├── verifier_agent.py      # Verifies solutions
│   └── explainer_agent.py     # Creates explanations
├── utils/                      # Utility modules
│   ├── ocr_handler.py         # Image to text
│   └── audio_handler.py       # Audio to text
├── rag/                        # RAG system
│   └── retriever.py           # Knowledge retrieval
└── memory/                     # Memory system
    └── simple_memory_handler.py
🤖 Agents Overview
1. Parser Agent
Extracts mathematical concepts from questions

Identifies problem type and difficulty

Tags relevant topics

2. Solver Agent
Uses Groq LLM for problem solving

Implements step-by-step solutions

Handles multiple mathematical domains

3. Verifier Agent
Checks solution correctness

Calculates confidence scores

Flags potential errors

4. Explainer Agent
Creates JEE-level explanations

Breaks down complex concepts

Provides learning tips

🎤 Audio Features
Supported Formats:
Upload: MP3, WAV, M4A, OGG, WEBM

Recording: Direct microphone input

Transcription: Google Speech Recognition

Tips for Better Audio:
Speak clearly and at normal pace

Use a good quality microphone

Minimize background noise

Keep recordings under 30 seconds

🖼️ OCR Features
Supported Image Formats:
JPG, JPEG, PNG

Tips for Better OCR:
Use clear, well-lit images

Typed text works better than handwritten

Avoid glare and shadows

Crop to the problem area

📊 Memory System
Stores problem-solution pairs

Learns from user feedback

Retrieves similar past problems

Tracks accuracy metrics

🔒 Security Notes
API Keys: Never commit .env files to version control

Sensitive Data: All user data is processed locally

Audio/Image Files: Temporary files are deleted after processing

🐛 Troubleshooting
Common Issues:
OCR Not Working:

Install Tesseract OCR

Check image quality

Verify file permissions

Audio Not Transcribing:

Install SpeechRecognition and pydub

Check microphone permissions

Ensure internet connection for Google API

LLM Not Responding:

Verify Groq API key in .env

Check internet connection

Confirm API quota not exceeded

🤝 Contributing
Fork the repository

Create a feature branch

Make your changes

Test thoroughly

Submit a pull request

📄 License
MIT License - see LICENSE file for details

🙏 Acknowledgements
Streamlit for the web framework

Groq for LLM API

SpeechRecognition for audio processing

Tesseract OCR for text extraction

📞 Support
For issues and questions:

Check the Troubleshooting section

Open an issue on GitHub

Contact the development team
