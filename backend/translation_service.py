"""
Sarvam AI Translation Service for v-askbot.
Handles all translation between Indic languages and English.
Sarvam is used ONLY for translation — Groq handles all RAG/research/answering.
"""
import time
import httpx
from collections import OrderedDict
from typing import Dict, Any
from backend.config import SARVAM_API_KEY

# Map short language codes to Sarvam language codes
LANGUAGE_CODE_MAP = {
    'en': 'en-IN',
    'hi': 'hi-IN',
    'bn': 'bn-IN',
    'te': 'te-IN',
    'ta': 'ta-IN',
    'mr': 'mr-IN',
    'gu': 'gu-IN',
    'kn': 'kn-IN',
    'ml': 'ml-IN',
    'pa': 'pa-IN',
    'or': 'od-IN',
    'ur': 'ur-IN',
    'as': 'as-IN',
    'ne': 'ne-IN',
    'sa': 'sa-IN',
    'sd': 'sd-IN',
    'kok': 'kok-IN',
}


class TranslationService:
    """
    Sarvam AI Translation Service.
    Uses the Sarvam Translate API (POST https://api.sarvam.ai/translate)
    to translate user queries to English and English answers back to user's language.
    """

    TRANSLATE_URL = "https://api.sarvam.ai/translate"
    MAX_CACHE_ENTRIES = 256
    MAX_INPUT_CHARS = 600

    def __init__(self, api_key: str = SARVAM_API_KEY):
        self.api_key = api_key
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(1.2, connect=0.35, read=0.8, write=0.35, pool=0.2),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        self._cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()

    def _get_sarvam_lang_code(self, lang: str) -> str:
        """Convert short lang code to Sarvam format."""
        return LANGUAGE_CODE_MAP.get(lang, f"{lang}-IN")

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Dict[str, Any]:
        """
        Translate text from source_lang to target_lang using Sarvam Translate API.
        Returns dict with 'translated_text', 'source_lang', 'target_lang', 'latency_ms'.
        """
        start_t = time.perf_counter()

        # If source == target, no translation needed
        if source_lang == target_lang:
            duration = (time.perf_counter() - start_t) * 1000
            return {
                'translated_text': text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'latency_ms': round(duration, 2),
                'skipped': True,
                'service': 'passthrough'
            }

        if not self.api_key or not text.strip():
            duration = (time.perf_counter() - start_t) * 1000
            return {
                'translated_text': text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'latency_ms': round(duration, 2),
                'skipped': True,
                'service': 'no_api_key'
            }

        source_code = self._get_sarvam_lang_code(source_lang)
        target_code = self._get_sarvam_lang_code(target_lang)
        cache_key = (text.strip(), source_lang, target_lang)
        cached_translation = self._cache.get(cache_key)
        if cached_translation is not None:
            self._cache.move_to_end(cache_key)
            return {
                'translated_text': cached_translation,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'latency_ms': 0.0,
                'skipped': True,
                'service': 'Sarvam AI Translate (cache)'
            }

        headers = {
            'api-subscription-key': self.api_key,
            'Content-Type': 'application/json'
        }
        payload = {
            'input': text[:self.MAX_INPUT_CHARS].strip(),
            'source_language_code': source_code,
            'target_language_code': target_code,
        }

        try:
            resp = await self._client.post(self.TRANSLATE_URL, headers=headers, json=payload)
            duration = (time.perf_counter() - start_t) * 1000

            if resp.status_code == 200:
                data = resp.json()
                translated = data.get('translated_text', text)
                self._cache[cache_key] = translated
                self._cache.move_to_end(cache_key)
                if len(self._cache) > self.MAX_CACHE_ENTRIES:
                    self._cache.popitem(last=False)
                return {
                    'translated_text': translated,
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'latency_ms': round(duration, 2),
                    'skipped': False,
                    'service': 'Sarvam AI Translate'
                }
            else:
                return {
                    'translated_text': text,
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'latency_ms': round(duration, 2),
                    'skipped': True,
                    'service': f'sarvam_error_{resp.status_code}',
                    'error': resp.text[:200]
                }
        except Exception as e:
            duration = (time.perf_counter() - start_t) * 1000
            return {
                'translated_text': text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'latency_ms': round(duration, 2),
                'skipped': True,
                'service': 'sarvam_exception',
                'error': str(e)[:200]
            }

    async def translate_to_english(self, text: str, source_lang: str) -> Dict[str, Any]:
        """Translate user's input from their language to English."""
        return await self.translate(text, source_lang, 'en')

    async def translate_from_english(self, text: str, target_lang: str) -> Dict[str, Any]:
        """Translate English answer back to user's language."""
        return await self.translate(text, 'en', target_lang)
