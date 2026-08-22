import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config import DATA_PATH

SUPPORTED_LANGUAGES = {
    'en': {'name': 'English', 'native': 'English', 'flag': 'ENG'},
    'hi': {'name': 'Hindi', 'native': 'हिन्दी', 'flag': 'HIN'},
    'bn': {'name': 'Bengali', 'native': 'বাংলা', 'flag': 'BEN'},
    'te': {'name': 'Telugu', 'native': 'తెలుగు', 'flag': 'TEL'},
    'ta': {'name': 'Tamil', 'native': 'தமிழ்', 'flag': 'TAM'},
    'mr': {'name': 'Marathi', 'native': 'मराठी', 'flag': 'MAR'},
    'gu': {'name': 'Gujarati', 'native': 'ગુજરાતી', 'flag': 'GUJ'},
    'kn': {'name': 'Kannada', 'native': 'ಕನ್ನಡ', 'flag': 'KAN'},
    'ml': {'name': 'Malayalam', 'native': 'മലയാളം', 'flag': 'MAL'},
    'pa': {'name': 'Punjabi', 'native': 'ਪੰਜਾਬੀ', 'flag': 'PUN'},
    'or': {'name': 'Odia', 'native': 'ଓଡ଼ିଆ', 'flag': 'ODI'}
}

class MSMARCOIndicLoader:
    def __init__(self, data_path: Path = DATA_PATH):
        self.data_path = Path(data_path)
        self.raw_data: List[Dict[str, Any]] = []
        self.load_data()

    def load_data(self) -> None:
        if self.data_path.exists():
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.raw_data = json.load(f)
        else:
            self.raw_data = []

    def get_all_documents(self, lang: Optional[str] = None) -> List[Dict[str, Any]]:
        docs = []
        for entry in self.raw_data:
            query_id = entry['query_id']
            domain = entry.get('domain', 'general')
            
            for l_code, l_data in entry.items():
                if l_code in ('query_id', 'domain'):
                    continue
                if lang and l_code != lang:
                    continue
                
                docs.append({
                    'doc_id': f"{query_id}_{l_code}",
                    'query_id': query_id,
                    'domain': domain,
                    'language': l_code,
                    'language_name': SUPPORTED_LANGUAGES.get(l_code, {}).get('name', l_code),
                    'query': l_data.get('query', ''),
                    'passage': l_data.get('passage', ''),
                    'answers': l_data.get('answers', [])
                })
        return docs

    def get_sample_queries(self, lang: str = "en") -> List[Dict[str, Any]]:
        samples = []
        for entry in self.raw_data:
            if lang in entry:
                samples.append({
                    'query_id': entry['query_id'],
                    'domain': entry.get('domain', 'general'),
                    'query': entry[lang]['query'],
                    'expected_answer': entry[lang]['answers'][0] if entry[lang].get('answers') else '',
                    'language': lang
                })
            elif 'en' in entry:
                samples.append({
                    'query_id': entry['query_id'],
                    'domain': entry.get('domain', 'general'),
                    'query': entry['en']['query'],
                    'expected_answer': entry['en']['answers'][0] if entry['en'].get('answers') else '',
                    'language': 'en'
                })
        return samples
