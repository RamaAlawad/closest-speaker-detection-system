"""
Closest Speaker Detection System - Complete Server Implementation
================================================================

This system detects which speaker is closest to the microphone by analyzing
audio intensity and voice activity detection (VAD) from multiple audio sources.

Features:
- Real-time audio processing from multiple clients
- Voice Activity Detection (WebRTC VAD or fallback volume detection)
- Audio intensity measurement in decibels
- Speech transcription (Google Speech API)
- WebSocket server for real-time communication
- Automatic speaker ranking by proximity/volume

Requirements:
    pip install websockets numpy webrtcvad SpeechRecognition

Author: Your Name
Date: 2025
Version: 1.0
"""

import asyncio
import json
import logging
import numpy as np
import websockets
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import base64
import io
import wave

import sys
import logging

# Force stdout/stderr to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)


# Optional imports with graceful fallbacks
try:
    import webrtcvad
    VAD_AVAILABLE = True
    print("✅ WebRTC VAD available")
except ImportError:
    VAD_AVAILABLE = False
    print("⚠️  WebRTC VAD not available - install with: pip install webrtcvad")

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
    print("✅ Speech Recognition available")
except ImportError:
    SR_AVAILABLE = False
    print("⚠️  Speech Recognition not available - install with: pip install SpeechRecognition")

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('speaker_detection.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SpeakerData:
    """
    Data class to store individual speaker information
    
    Attributes:
        speaker_id (str): Unique identifier for the speaker
        intensity_db (float): Audio intensity in decibels (-60 to 0)
        is_speaking (bool): Whether speaker is currently speaking
        last_transcription (str): Most recent speech transcription
        last_update (datetime): Timestamp of last audio update
        audio_buffer (List[bytes]): Rolling buffer of recent audio chunks
        connection_time (datetime): When speaker first connected
        total_speech_time (float): Total seconds of detected speech
    """
    speaker_id: str
    intensity_db: float
    is_speaking: bool
    last_transcription: str
    last_update: datetime
    audio_buffer: List[bytes]
    connection_time: datetime
    total_speech_time: float = 0.0

class AudioProcessor:
    """
    Handles all audio processing tasks including VAD, intensity measurement, and transcription
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the audio processor
        
        Args:
            sample_rate (int): Audio sample rate in Hz (default: 16000)
        """
        self.sample_rate = sample_rate
        self.vad = None
        self.recognizer = None
        
        # Audio processing parameters
        self.vad_frame_duration_ms = 30  # VAD frame duration in milliseconds
        self.speech_threshold_db = -45.0  # dB threshold for speech detection
        self.silence_timeout = 2.0  # Seconds of silence before stopping transcription
        
        self._initialize_vad()
        self._initialize_speech_recognition()
        
        logger.info(f"AudioProcessor initialized (sample_rate={sample_rate}Hz)")
        
    def _initialize_vad(self) -> None:
        """Initialize WebRTC Voice Activity Detection"""
        if VAD_AVAILABLE:
            try:
                # Aggressiveness levels: 0=least aggressive, 3=most aggressive
                self.vad = webrtcvad.Vad(2)
                logger.info("WebRTC VAD initialized with aggressiveness level 2")
            except Exception as e:
                logger.warning(f"Failed to initialize WebRTC VAD: {e}")
                self.vad = None
        
    def _initialize_speech_recognition(self) -> None:
        """Initialize Google Speech Recognition"""
        if SR_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                # Adjust recognition sensitivity
                self.recognizer.energy_threshold = 300
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 0.8
                logger.info("Speech Recognition initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Speech Recognition: {e}")
                self.recognizer = None
    
    def calculate_rms_db(self, audio_data: bytes) -> float:
        """
        Calculate Root Mean Square (RMS) amplitude and convert to decibels
        
        Args:
            audio_data (bytes): Raw 16-bit PCM audio data
            
        Returns:
            float: Audio intensity in decibels (-60 to 0 dB range)
        """
        try:
            # Convert bytes to numpy array (16-bit signed integers)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            if len(audio_array) == 0:
                return -60.0  # Minimum dB level for silence
            
            # Calculate RMS (Root Mean Square)
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            
            # Convert to decibels relative to maximum 16-bit value
            if rms > 0:
                db = 20 * np.log10(rms / 32767.0)
                return max(db, -60.0)  # Clamp to minimum -60 dB
            else:
                return -60.0
                
        except Exception as e:
            logger.error(f"Error calculating RMS: {e}")
            return -60.0
    
    def detect_speech(self, audio_data: bytes) -> bool:
        """
        Detect speech in audio using VAD or volume-based detection
        
        Args:
            audio_data (bytes): Raw 16-bit PCM audio data
            
        Returns:
            bool: True if speech is detected, False otherwise
        """
        try:
            # Method 1: Try WebRTC VAD first (more accurate)
            if self.vad is not None:
                return self._vad_detect_speech(audio_data)
            
            # Method 2: Fallback to simple volume-based detection
            return self._volume_detect_speech(audio_data)
            
        except Exception as e:
            logger.error(f"Error in speech detection: {e}")
            return False
    
    def _vad_detect_speech(self, audio_data: bytes) -> bool:
        """
        Use WebRTC VAD for speech detection
        
        Args:
            audio_data (bytes): Raw audio data
            
        Returns:
            bool: True if speech detected
        """
        try:
            # WebRTC VAD requires specific frame sizes
            frame_size = int(self.sample_rate * self.vad_frame_duration_ms / 1000) * 2  # 2 bytes per sample
            
            # Process audio in frames
            speech_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_data), frame_size):
                frame = audio_data[i:i + frame_size]
                
                # Skip incomplete frames
                if len(frame) != frame_size:
                    continue
                    
                total_frames += 1
                if self.vad.is_speech(frame, self.sample_rate):
                    speech_frames += 1
            
            # Consider it speech if more than 30% of frames contain speech
            if total_frames > 0:
                speech_ratio = speech_frames / total_frames
                return speech_ratio > 0.3
                
            return False
            
        except Exception as e:
            logger.debug(f"VAD error, falling back to volume detection: {e}")
            return self._volume_detect_speech(audio_data)
    
    def _volume_detect_speech(self, audio_data: bytes) -> bool:
        """
        Simple volume-based speech detection
        
        Args:
            audio_data (bytes): Raw audio data
            
        Returns:
            bool: True if volume exceeds threshold
        """
        intensity_db = self.calculate_rms_db(audio_data)
        return intensity_db > self.speech_threshold_db
    
    def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """
        Transcribe audio to text using Google Speech API
        
        Args:
            audio_data (bytes): Raw 16-bit PCM audio data
            
        Returns:
            Optional[str]: Transcribed text or None if transcription failed
        """
        if not self.recognizer:
            return None
            
        try:
            # Create WAV file in memory
            audio_io = io.BytesIO()
            with wave.open(audio_io, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)
            
            audio_io.seek(0)
            
            # Transcribe using Google Speech API
            with sr.AudioFile(audio_io) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio = self.recognizer.record(source)
                
                # Attempt transcription
                text = self.recognizer.recognize_google(
                    audio, 
                    language='en-US',
                    show_all=False
                )
                
                return text.strip() if text else None
                
        except sr.UnknownValueError:
            # Speech was unintelligible
            logger.debug("Speech was unintelligible")
            return None
        except sr.RequestError as e:
            # API request failed
            logger.error(f"Speech recognition API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

class SpeakerManager:
    """
    Manages multiple speakers and determines the closest/loudest speaker
    """
    
    def __init__(self):
        """Initialize the speaker manager"""
        self.speakers: Dict[str, SpeakerData] = {}
        self.audio_processor = AudioProcessor()
        self.lock = threading.Lock()
        
        # Configuration
        self.max_speaker_timeout = 30  # Remove inactive speakers after 30 seconds
        self.audio_buffer_size = 30    # Keep 30 chunks (~3 seconds at 100ms chunks)
        self.transcription_cooldown = 3  # Wait 3 seconds between transcriptions per speaker
        
        # Statistics
        self.total_processed_chunks = 0
        self.start_time = datetime.now()
        
        logger.info("SpeakerManager initialized")
        
    def update_speaker_audio(self, speaker_id: str, audio_data: bytes) -> None:
        """
        Update a speaker's audio data and process it
        
        Args:
            speaker_id (str): Unique identifier for the speaker
            audio_data (bytes): Raw 16-bit PCM audio data
        """
        with self.lock:
            current_time = datetime.now()
            
            # Initialize new speaker
            if speaker_id not in self.speakers:
                self.speakers[speaker_id] = SpeakerData(
                    speaker_id=speaker_id,
                    intensity_db=-60.0,
                    is_speaking=False,
                    last_transcription="",
                    last_update=current_time,
                    audio_buffer=[],
                    connection_time=current_time
                )
                logger.info(f"New speaker registered: {speaker_id}")
            
            speaker = self.speakers[speaker_id]
            
            # Process audio
            speaker.intensity_db = self.audio_processor.calculate_rms_db(audio_data)
            was_speaking = speaker.is_speaking
            speaker.is_speaking = self.audio_processor.detect_speech(audio_data)
            speaker.last_update = current_time
            
            # Track speech time
            if speaker.is_speaking and not was_speaking:
                logger.debug(f"{speaker_id} started speaking")
            elif was_speaking and not speaker.is_speaking:
                logger.debug(f"{speaker_id} stopped speaking")
                
            if speaker.is_speaking:
                speaker.total_speech_time += 0.1  # Assuming 100ms chunks
            
            # Manage audio buffer
            speaker.audio_buffer.append(audio_data)
            if len(speaker.audio_buffer) > self.audio_buffer_size:
                speaker.audio_buffer.pop(0)
                
            self.total_processed_chunks += 1
            
    def get_closest_speaker(self) -> Optional[SpeakerData]:
        """
        Get the speaker with the highest audio intensity who is currently speaking
        
        Returns:
            Optional[SpeakerData]: The closest speaking speaker or None
        """
        with self.lock:
            current_time = datetime.now()
            
            # Filter for active, speaking speakers
            speaking_speakers = [
                speaker for speaker in self.speakers.values()
                if speaker.is_speaking and 
                   (current_time - speaker.last_update).total_seconds() < 5
            ]
            
            if not speaking_speakers:
                return None
                
            # Return speaker with highest intensity
            return max(speaking_speakers, key=lambda s: s.intensity_db)
    
    def get_ranked_speakers(self) -> List[SpeakerData]:
        """
        Get all active speakers ranked by audio intensity
        
        Returns:
            List[SpeakerData]: Speakers sorted by intensity (loudest first)
        """
        with self.lock:
            current_time = datetime.now()
            
            # Filter active speakers (updated within last 10 seconds)
            active_speakers = [
                speaker for speaker in self.speakers.values()
                if (current_time - speaker.last_update).total_seconds() < 10
            ]
            
            # Sort by intensity (highest first)
            return sorted(active_speakers, key=lambda s: s.intensity_db, reverse=True)
    
    async def transcribe_closest_speaker(self) -> Optional[str]:
        """
        Transcribe the closest speaker's recent audio
        
        Returns:
            Optional[str]: Transcribed text or None
        """
        closest_speaker = self.get_closest_speaker()
        if not closest_speaker or not closest_speaker.audio_buffer:
            return None
            
        # Check transcription cooldown
        if closest_speaker.last_transcription:
            time_since_last = (datetime.now() - closest_speaker.last_update).total_seconds()
            if time_since_last < self.transcription_cooldown:
                return None
        
        # Combine recent audio chunks (last 1 second)
        recent_chunks = closest_speaker.audio_buffer[-10:]  # Last 10 chunks
        combined_audio = b''.join(recent_chunks)
        
        if len(combined_audio) < 1600:  # Minimum audio length (0.1 second at 16kHz)
            return None
        
        # Transcribe in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            transcription = await loop.run_in_executor(
                None, 
                self.audio_processor.transcribe_audio, 
                combined_audio
            )
            
            if transcription:
                closest_speaker.last_transcription = transcription
                logger.info(f"Transcription from {closest_speaker.speaker_id}: {transcription}")
                
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    def cleanup_inactive_speakers(self) -> None:
        """Remove speakers that haven't sent audio recently"""
        with self.lock:
            current_time = datetime.now()
            inactive_speakers = [
                speaker_id for speaker_id, speaker in self.speakers.items()
                if (current_time - speaker.last_update).total_seconds() > self.max_speaker_timeout
            ]
            
            for speaker_id in inactive_speakers:
                logger.info(f"Removing inactive speaker: {speaker_id}")
                del self.speakers[speaker_id]
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        with self.lock:
            uptime = (datetime.now() - self.start_time).total_seconds()
            return {
                'total_speakers': len(self.speakers),
                'active_speakers': len(self.get_ranked_speakers()),
                'total_processed_chunks': self.total_processed_chunks,
                'uptime_seconds': uptime,
                'chunks_per_second': self.total_processed_chunks / max(uptime, 1)
            }

class WebSocketServer:
    """
    WebSocket server for handling real-time communication with clients
    """
    
    def __init__(self, host: str = 'localhost', port: int = 8765):
        """
        Initialize WebSocket server
        
        Args:
            host (str): Server host address
            port (int): Server port number
        """
        self.host = host
        self.port = port
        self.speaker_manager = SpeakerManager()
        self.clients = set()
        
        # Server configuration
        self.max_message_size = 10**7  # 10MB max message size
        self.ping_interval = 20        # Ping clients every 20 seconds
        self.ping_timeout = 10         # Timeout after 10 seconds
        
        logger.info(f"WebSocket server initialized on {host}:{port}")
        
    async def handle_client(self, websocket):
        """
        Handle individual client connections
        
        Args:
            websocket: WebSocket connection object
            path: Connection path (unused)
        """
        # Generate unique client ID
        client_id = f"speaker_{len(self.clients) + 1}_{int(time.time())}"
        self.clients.add(websocket)
        
        # Store client info in websocket
        websocket.client_id = client_id
        websocket.connect_time = datetime.now()
        
        logger.info(f"Client connected: {client_id} (Total clients: {len(self.clients)})")
        
        try:
            # Send welcome message
            await self.send_welcome_message(websocket, client_id)
            
            # Handle incoming messages
            async for message in websocket:
                await self.process_message(message, client_id, websocket)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Client error ({client_id}): {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client removed: {client_id} (Remaining: {len(self.clients)})")
    
    async def send_welcome_message(self, websocket, client_id: str):
        """Send welcome message to new client"""
        welcome_msg = {
            'type': 'welcome',
            'client_id': client_id,
            'server_info': {
                'vad_available': VAD_AVAILABLE and self.speaker_manager.audio_processor.vad is not None,
                'sr_available': SR_AVAILABLE and self.speaker_manager.audio_processor.recognizer is not None,
                'sample_rate': self.speaker_manager.audio_processor.sample_rate,
                'server_time': datetime.now().isoformat()
            }
        }
        await websocket.send(json.dumps(welcome_msg))
    
    async def process_message(self, message, client_id: str, websocket):
        """
        Process incoming messages from clients
        
        Args:
            message: Raw message data (JSON or binary)
            client_id: Client identifier
            websocket: WebSocket connection
        """
        try:
            # Try to parse as JSON first
            if isinstance(message, str):
                data = json.loads(message)
                await self.process_json_message(data, client_id, websocket)
            else:
                # Handle binary audio data
                await self.process_audio_data(message, client_id, websocket)
                
        except json.JSONDecodeError:
            # If not JSON, treat as binary audio
            await self.process_audio_data(message, client_id, websocket)
        except Exception as e:
            logger.error(f"Error processing message from {client_id}: {e}")
            await self.send_error(websocket, f"Message processing error: {e}")
     
    async def process_json_message(self, data: dict, client_id: str, websocket):
        """
        Process JSON command messages
        
        Args:
            data: Parsed JSON data
            client_id: Client identifier  
            websocket: WebSocket connection
        """
        message_type = data.get('type')
        
        if message_type == 'audio_data':
            # Base64 encoded audio data
            audio_b64 = data.get('audio')
            if audio_b64:
                try:
                    audio_data = base64.b64decode(audio_b64)
                    await self.process_audio_data(audio_data, client_id, websocket)
                except Exception as e:
                    logger.error(f"Error decoding base64 audio from {client_id}: {e}")
        
        elif message_type == 'get_status':
            # Send current status
            await self.send_status_update(websocket)
            
        elif message_type == 'get_statistics':
            # Send system statistics
            await self.send_statistics(websocket)
            
        elif message_type == 'ping':
            # Respond to ping
            await websocket.send(json.dumps({
                'type': 'pong', 
                'timestamp': time.time(),
                'client_id': client_id
            }))
            
        else:
            logger.warning(f"Unknown message type from {client_id}: {message_type}")
    
    async def process_audio_data(self, audio_data: bytes, client_id: str, websocket):
        """
        Process binary audio data
        
        Args:
            audio_data: Raw audio bytes
            client_id: Client identifier
            websocket: WebSocket connection
        """
        try:
            # Update speaker data
            self.speaker_manager.update_speaker_audio(client_id, audio_data)
            
            # Trigger transcription asynchronously (don't block)
            asyncio.create_task(self.transcribe_and_broadcast())
            
        except Exception as e:
            logger.error(f"Error processing audio from {client_id}: {e}")
    
    async def transcribe_and_broadcast(self):
        """Transcribe closest speaker and broadcast updates"""
        try:
            # Attempt transcription
            await self.speaker_manager.transcribe_closest_speaker()
            
            # Broadcast updated status to all clients
            await self.broadcast_status()
            
        except Exception as e:
            logger.error(f"Error in transcribe_and_broadcast: {e}")
    
    
       
    def safe_json(data):
         if isinstance(data, dict):
              return {k: safe_json(v) for k, v in data.items()}
         elif isinstance(data, list):
               return [safe_json(v) for v in data]
         elif isinstance(data, (np.float32, np.float64)):
              return float(data)
         elif isinstance(data, (np.int32, np.int64)):
              return int(data)
         else:
            return data
    async def send_status_update(self, websocket):
        """
        Send current system status to a specific client
        
        Args:
            websocket: Target WebSocket connection
        """
        try:
            ranked_speakers = self.speaker_manager.get_ranked_speakers()
            closest_speaker = self.speaker_manager.get_closest_speaker()
            
            status = {
                'type': 'status_update',
                'timestamp': datetime.now().isoformat(),
                'closest_speaker': closest_speaker.speaker_id if closest_speaker else None,
                'speakers': [
                    {
                        'id': speaker.speaker_id,
                        'intensity_db': float(round(speaker.intensity_db, 2)),
                        'is_speaking': speaker.is_speaking,
                        'last_transcription': speaker.last_transcription,
                        'rank': idx + 1,
                        'connection_time': speaker.connection_time.isoformat(),
                        'total_speech_time': float(round(speaker.total_speech_time, 1))
                    }
                    for idx, speaker in enumerate(ranked_speakers)
                ],
                'server_info': {
                    'vad_available': VAD_AVAILABLE and self.speaker_manager.audio_processor.vad is not None,
                    'sr_available': SR_AVAILABLE and self.speaker_manager.audio_processor.recognizer is not None,
                    'active_speakers': len(ranked_speakers),
                    'total_clients': len(self.clients)
                }
            }
            
            await websocket.send(json.dumps(status))
            
        except websockets.exceptions.ConnectionClosed:
            # Client disconnected, ignore
            pass
        except Exception as e:
            logger.error(f"Error sending status update: {e}")
    
    async def send_statistics(self, websocket):
        """Send system statistics to client"""
        try:
            stats = self.speaker_manager.get_statistics()
            message = {
                'type': 'statistics',
                'timestamp': datetime.now().isoformat(),
                'statistics': stats
            }
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending statistics: {e}")
    
    async def send_error(self, websocket, error_message: str):
        """Send error message to client"""
        try:
            error_msg = {
                'type': 'error',
                'message': error_message,
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(error_msg))
        except:
            pass  # Ignore errors when sending errors
    
    async def broadcast_status(self):
        """Broadcast current status to all connected clients"""
        if not self.clients:
            return
            
        # Collect disconnected clients
        disconnected = set()
        
        # Send to all clients
        for websocket in self.clients.copy():
            try:
                await self.send_status_update(websocket)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.clients -= disconnected
    
    async def start_server(self):
        """Start the WebSocket server and background tasks"""
        logger.info(f"🚀 Starting WebSocket server on {self.host}:{self.port}")
        
        # Start background tasks
        asyncio.create_task(self.periodic_updates())
        asyncio.create_task(self.cleanup_task())
        
        try:
            # Start WebSocket server
            async with websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                max_size=self.max_message_size,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout,
                
            ):
                logger.info("✅ Server started successfully!")
                logger.info(f"🌐 WebSocket URL: ws://{self.host}:{self.port}")
                logger.info("📱 Open the HTML client in your browser to connect")
                logger.info("🔧 Press Ctrl+C to stop the server")
                
                # Run forever
                await asyncio.Future()
                
        except OSError as e:
            if "Address already in use" in str(e):
                logger.error(f"❌ Port {self.port} is already in use. Try a different port.")
            else:
                logger.error(f"❌ Server startup error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected server error: {e}")
            raise
    
    async def periodic_updates(self):
        """Send periodic status updates to all clients"""
        while True:
            try:
                await asyncio.sleep(1)  # Update every second
                await self.broadcast_status()
            except Exception as e:
                logger.error(f"Periodic update error: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    async def cleanup_task(self):
        """Periodic cleanup of inactive speakers"""
        while True:
            try:
                await asyncio.sleep(30)  # Cleanup every 30 seconds
                self.speaker_manager.cleanup_inactive_speakers()
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(60)

def main():
    """
    Main function to start the server
    """
    print("🎤 Closest Speaker Detection System")
    print("=" * 60)
    print(f"🔧 WebRTC VAD: {'✅ Available' if VAD_AVAILABLE else '❌ Not Available (install webrtcvad)'}")
    print(f"🔧 Speech Recognition: {'✅ Available' if SR_AVAILABLE else '❌ Not Available (install SpeechRecognition)'}")
    print(f"🔧 Python Version: {'.'.join(map(str, __import__('sys').version_info[:3]))}")
    print("=" * 60)
    
    # Create and configure server
    server = WebSocketServer(host='localhost', port=8765)
    
    try:
        # Run the server
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        logger.info("Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        logger.error(f"Server error: {e}")
    finally:
        print("🔚 Server shutdown complete")

if __name__ == "__main__":
    main()