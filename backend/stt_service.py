import os
import time
import httpx
from typing import Dict, Any
from backend.config import SARVAM_API_KEY, ELEVENLABS_API_KEY

class SpeechToTextService:
    def __init__(self, sarvam_key: str = SARVAM_API_KEY, eleven_key: str = ELEVENLABS_API_KEY):
        self.sarvam_key = sarvam_key
        self.eleven_key = eleven_key

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = 'audio.wav',
        language_code: str = 'hi-IN'
    ) -> Dict[str, Any]:
        start_t = time.perf_counter()

        if self.sarvam_key and len(audio_bytes) > 100:
            try:
                headers = {'api-subscription-key': self.sarvam_key}
                files = {'file': (filename, audio_bytes, 'audio/wav')}
                data = {'model': 'saaras:v3', 'language_code': language_code}

                async with httpx.AsyncClient(timeout=4.0) as client:
                    resp = await client.post('https://api.sarvam.ai/speech-to-text', headers=headers, files=files, data=data)
                    if resp.status_code == 200:
                        res_json = resp.json()
                        duration = (time.perf_counter() - start_t) * 1000
                        return {
                            'transcript': res_json.get('transcript', ''),
                            'language_code': res_json.get('language_code', language_code),
                            'service': 'Sarvam AI (saaras:v3)',
                            'latency_ms': round(duration, 2)
                        }
            except Exception:
                pass

        duration = (time.perf_counter() - start_t) * 1000
        return {
            'transcript': '',
            'language_code': language_code,
            'service': 'unavailable',
            'latency_ms': round(duration, 2),
            'error': 'Sarvam STT is unavailable or no valid audio was received.'
        }
