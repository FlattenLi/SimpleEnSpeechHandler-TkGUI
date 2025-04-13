# SimpleEnSpeechHandler-TkGUI
An python app for dealing English speech materials powered by tkinter and Whisper. It can autoly recognize text, mark timestamp and craft a listening ppt.


**Core Functionality**  
*Your Audio Processing Toolkit*  

1. **Audio Analysis**  
   - <u>Automatic text & timestamp recognition in audio files</u>  
   - Manual calibration for precision alignment  
   - Video file support (processed as audio)  

2. **Content Management**  
   - Copy individual sentence texts  
   - Save articles as `.txt` files  
   - Export audio clips per sentence  

3. **Advanced Customization**  
   - <u>Audio crafting with custom</u>:  
     ⮞ Repeat patterns  
     ⮞ Separation parameters  
     ⮞ Sorting logic  
     ⮞ Speed adjustments  
   - Waveform visualization  

4. **Special Features**  
   - <u>Generate listening-focused PowerPoint presentations</u>  
   - *Additional features in development...*  

---

**User Guide**  
*Essential Interaction Tips*  

- **Sentence Controls**  
  ▶ Left-click: Select sentence & reveal options  
  ▶ Right-click: Context menu for selected text  

- **Navigation Tools**  
  🔍 Text Viewer: Access via topbar 'Text' button  
  🎚️ Audio Board: Real-time sentence highlighting  

*Common options excluded from basic documentation*

---

**System Requirements**  
*Technical Specifications*  

- **Python 3.11+**  
  *(Match statement dependency - downgrade possible with if-else refactoring)*  

- **Core Libraries**:  
  ▸ `whisper` (base.en recommended, >200MB)  
  ▸ `transformers` (model operations)  
  ▸ `matplotlib` (waveform visualization)  

- **Media Handlers**:  
  ▸ `pydub` | `pyaudio` | `moviepy`  

- **Office Integration**:  
  ▸ `python-pptx` (PPT generation)  

---

**Roadmap & Updates**  
*What's Coming Next*  

- **Next Release (ETA: 1 Month)**  
  ▶ Final feature implementation  
  ▶ Standalone executable build  
  ▶ Comprehensive environment setup guide  

- **Documentation Improvements**  
  ▶ Complete README.md overhaul  
  ▶ User-friendly installation instructions  
  ▶ Detailed feature breakdown  

---
This text is beautified by deepseek-r1.
