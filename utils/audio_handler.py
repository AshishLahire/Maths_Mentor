"""
SIMPLE Audio Handler for Math Mentor AI
Works with current SpeechRecognition version
"""

import tempfile
import os
import wave
import speech_recognition as sr

class SimpleAudioTranscriber:
    """
    Simple audio transcription class
    """
    
    def __init__(self):
        """Initialize simple audio transcription"""
        print("🎤 Initializing Simple Audio Transcriber...")
        
        try:
            self.recognizer = sr.Recognizer()
            print("✅ SpeechRecognition: Loaded")
            
            # Check if AudioFile is available
            if hasattr(sr, 'AudioFile'):
                print("✅ AudioFile: Available")
            else:
                print("⚠️ AudioFile: Not in sr module, checking alternatives")
                
        except ImportError:
            print("❌ SpeechRecognition: Not installed")
            print("   Install: pip install SpeechRecognition")
            self.recognizer = None
        except Exception as e:
            print(f"⚠️ SpeechRecognition error: {e}")
            self.recognizer = None
    
    def transcribe_audio_file(self, audio_path):
        """
        Transcribe an audio file to text
        Returns: (text, confidence)
        """
        if not self.recognizer:
            return "Speech recognition not available. Install SpeechRecognition.", 0.1
        
        try:
            print(f"🔊 Processing: {os.path.basename(audio_path)}")
            
            # Check if file exists
            if not os.path.exists(audio_path):
                return f"Audio file not found: {audio_path}", 0.0
            
            # Get file size
            file_size = os.path.getsize(audio_path)
            print(f"   Size: {file_size:,} bytes")
            
            # Try to transcribe
            with sr.AudioFile(audio_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Record the audio
                audio = self.recognizer.record(source)
                
                # Recognize using Google Web Speech API
                text = self.recognizer.recognize_google(audio, language='en-US')
                
                # Calculate confidence
                words = text.split()
                word_count = len(words)
                
                if word_count == 0:
                    confidence = 0.1
                elif word_count < 3:
                    confidence = 0.3
                elif word_count < 6:
                    confidence = 0.5
                elif word_count < 10:
                    confidence = 0.7
                else:
                    confidence = 0.85
                
                print(f"   ✅ Transcription successful")
                print(f"   📝 Words: {word_count} | Confidence: {confidence:.0%}")
                
                return text.strip(), confidence
                
        except sr.UnknownValueError:
            msg = "Could not understand audio. Please speak more clearly."
            print(f"   ❌ {msg}")
            return msg, 0.2
            
        except sr.RequestError as e:
            msg = f"Speech service error: {e}"
            print(f"   ❌ {msg}")
            return msg, 0.1
            
        except Exception as e:
            msg = f"Audio processing error: {str(e)}"
            print(f"   ❌ {msg}")
            return msg, 0.1
    
    def transcribe_recording(self, audio_bytes, sample_rate=16000):
        """
        Transcribe audio from microphone recording bytes
        Returns: (text, confidence)
        """
        if not self.recognizer:
            return "Speech recognition not available.", 0.1
        
        # Save bytes to temp file
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_wav.name
        
        try:
            # Write audio bytes to WAV file
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)
            
            # Transcribe the file
            return self.transcribe_audio_file(temp_path)
            
        except Exception as e:
            msg = f"Recording error: {str(e)}"
            return msg, 0.1
        
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

# Global instance
_transcriber = None

def get_transcriber():
    """Get global transcriber instance"""
    global _transcriber
    if _transcriber is None:
        _transcriber = SimpleAudioTranscriber()
    return _transcriber

def transcribe_audio(audio_path):
    """Transcribe audio file (public function)"""
    transcriber = get_transcriber()
    return transcriber.transcribe_audio_file(audio_path)

def transcribe_recording(audio_bytes, sample_rate=16000):
    """Transcribe recorded audio (public function)"""
    transcriber = get_transcriber()
    return transcriber.transcribe_recording(audio_bytes, sample_rate)

# Test
if __name__ == "__main__":
    print("\n🧪 Testing Simple Audio System...")
    
    # Create test audio file
    import numpy as np
    
    test_file = "test_simple.wav"
    duration = 2.0
    sample_rate = 16000
    
    try:
        # Generate tone
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * 440 * t) * 0.5
        
        # Convert to 16-bit PCM
        audio_data = (tone * 32767).astype(np.int16)
        
        # Save as WAV
        with wave.open(test_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        print(f"✅ Created test file: {test_file}")
        
        # Test
        transcriber = get_transcriber()
        if transcriber.recognizer:
            text, confidence = transcriber.transcribe_audio_file(test_file)
            print(f"📝 Result: {text}")
            print(f"📊 Confidence: {confidence:.0%}")
        else:
            print("❌ Speech recognition not available")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        if os.path.exists(test_file):
            try:
                os.unlink(test_file)
            except:
                pass
