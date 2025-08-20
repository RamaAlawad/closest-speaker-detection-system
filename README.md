  # ***🎤 Closest Speaker Detection System*** #
This project is a complete system for detecting the closest speaker to a microphone by analyzing audio intensity and voice activity from multiple audio sources. It consists of a Python-based WebSocket server and a a web-based client interface.

🚀 Features
Real-time Audio Processing: The system handles real-time audio streams from multiple clients concurrently.

Voice Activity Detection (VAD): It uses the WebRTC VAD library for efficient voice detection, with a simple volume-based fallback if the library is not available.

Audio Intensity Measurement: It measures audio intensity in decibels to accurately rank speakers by their proximity to the microphone.

Speech Transcription: It integrates with the Google Speech API to transcribe spoken words into text.

WebSocket Communication: A WebSocket server is used for real-time, low-latency communication between the clients and the backend.

Responsive Web Interface: A modern and responsive web interface is provided for easy interaction and visualization of the results.

🛠️ Installation
You will need Python 3.x installed on your system.

Clone the repository from GitHub:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Install dependencies using pip:

pip install -r requirements.txt

▶️ Usage
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
