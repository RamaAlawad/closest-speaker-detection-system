# 🎤 Closest Speaker Detection System

![Python](https://img.shields.io/badge/Python-3.x-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![GitHub Pages](https://img.shields.io/badge/GitHub-Pages-orange)

A complete system for detecting the closest speaker to a microphone by analyzing audio intensity and voice activity from multiple audio sources. It consists of a Python-based WebSocket server and a web-based client interface.

---

## 📌 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [File Descriptions](#-file-descriptions)
- [License](#-license)
- [Contributing](#-contributing)
- [Live Demo](#-live-demo)

---

## 🚀 Features

- **Real-time Audio Processing:** Handles audio streams from multiple clients concurrently.  
- **Voice Activity Detection (VAD):** Uses the WebRTC VAD library, with a volume-based fallback.  
- **Audio Intensity Measurement:** Measures audio intensity in decibels to rank speakers by proximity.  
- **Speech Transcription:** Integrates with Google Speech API to convert speech to text.  
- **WebSocket Communication:** Low-latency, real-time communication between backend and clients.  
- **Responsive Web Interface:** Modern UI for interaction and visualization of results.  

---

## 🛠️ Installation

1. **Prerequisites:**  
   - Python 3.x installed on your system

2. **Clone the repository:**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

3. Install dependencies using pip:
```
pip install -r requirements.txt
```
## ▶️ Usage
Start the server:

python main.py

The server will start on localhost at port 8765.

Open the client interface:
Open the index.html file in your web browser. You can do this by simply double-clicking the file or by serving it via a local web server (e.g., python -m http.server).

Connect and Record:

In the web interface, click the "Connect Server" button. Your browser will prompt you to grant microphone permissions.

Once connected, click "Start Recording" to begin the audio processing.

The web interface will show a list of connected speakers, their audio levels, and highlight the speaker who is currently closest to the microphone.

📄 File Descriptions
main.py: The core backend server implementation written in Python. It handles WebSocket connections, audio processing, VAD, and transcription.

index.html: The frontend web interface that allows users to connect to the server, record audio, and view the real-time speaker detection results.

requirements.txt: A list of all Python libraries required to run the server.

📝 License
This project is open source and available under the MIT License. See the LICENSE file for more details.

🤝 Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests.
