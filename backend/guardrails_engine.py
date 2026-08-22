import re
from typing import Dict, Any, List, Optional
from backend.config import GUARDRAIL_GROUNDEDNESS_THRESHOLD, SIMILARITY_THRESHOLD

class GuardrailsEngine:
    INJECTION_PATTERNS = [
        r'ignore (all )?previous instructions',
        r'disregard (all )?guidelines',
        r'you are now in dan mode',
        r'system prompt',
        r'override security',
        r'jailbreak',
        r'reveal (your )?secret'
    ]

    TOXIC_KEYWORDS = [
        'kill', 'bomb', 'terrorist', 'hack into', 'credit card fraud',
        'chutiya', 'madarchod', 'harami', 'bokachoda', 'thevidiya'
    ]

    def check_input(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower().strip()
        
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, q_lower):
                return {
                    'passed': False,
                    'reason': 'Prompt Injection Detected',
                    'action': 'BLOCK_INPUT',
                    'clean_query': query
                }

        for kw in self.TOXIC_KEYWORDS:
            if kw in q_lower:
                return {
                    'passed': False,
                    'reason': 'Inappropriate / Toxic content detected',
                    'action': 'BLOCK_INPUT',
                    'clean_query': query
                }

        sanitized = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[REDACTED_EMAIL]', query)
        sanitized = re.sub(r'\b\d{10}\b', '[REDACTED_PHONE]', sanitized)

        return {
            'passed': True,
            'reason': 'Input validated successfully',
            'action': 'ALLOW',
            'clean_query': sanitized
        }

    def check_retrieval_sufficiency(self, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not retrieved_chunks:
            return {
                'sufficient': False,
                'top_score': 0.0,
                'action': 'TRIGGER_REFUSAL',
                'reason': 'No relevant knowledge found in MSMARCO-XI dataset'
            }

        top_score = retrieved_chunks[0]['score']
        if top_score < SIMILARITY_THRESHOLD:
            return {
                'sufficient': False,
                'top_score': top_score,
                'action': 'TRIGGER_REFUSAL',
                'reason': f"Low retrieval relevance score ({top_score:.2f})"
            }

        top_chunk = retrieved_chunks[0]
        required_overlap = min(2, top_chunk.get('query_keyword_count', 2))
        if top_chunk.get('keyword_overlap', 0) < required_overlap:
            return {
                'sufficient': False,
                'top_score': top_score,
                'action': 'TRIGGER_REFUSAL',
                'reason': 'Vector similarity lacked enough exact query-term overlap'
            }

        return {
            'sufficient': True,
            'top_score': top_score,
            'action': 'PROCEED',
            'reason': f"Context sufficiency validated (top score: {top_score:.3f})"
        }

    def check_output_groundedness(self, answer: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not answer or not context_chunks:
            return {
                'grounded': False,
                'groundedness_score': 0.0,
                'hallucination_detected': True,
                'verdict': 'Refusal: Insufficient verified context'
            }

        combined_context = ' '.join([c.get('text', '') + ' ' + c.get('effective_context', '') for c in context_chunks]).lower()
        ans_tokens = [w for w in re.findall(r'\w+', answer.lower()) if len(w) > 3]

        if not ans_tokens:
            return {'grounded': True, 'groundedness_score': 1.0, 'hallucination_detected': False, 'verdict': 'Short response'}

        matches = sum(1 for token in ans_tokens if token in combined_context)
        score = min(1.0, (matches / len(ans_tokens)) * 1.30)
        is_grounded = score >= GUARDRAIL_GROUNDEDNESS_THRESHOLD

        return {
            'grounded': is_grounded,
            'groundedness_score': round(score, 3),
            'hallucination_detected': not is_grounded,
            'verdict': 'PASSED_GROUNDED' if is_grounded else 'FAILED_HALLUCINATION_RISK'
        }
    def generate_refusal_message(self, language: str = 'en') -> str:
        refusals = { 
            'en': 'I do not have sufficient verified context in the MSMARCO-XI dataset to answer this question accurately.',
            'hi': 'मेरे पास इस प्रश्न का सटीक उत्तर देने के लिए MSMARCO-XI डेटासेट में पर्याप्त सत्यापित संदर्भ नहीं है।',
            'bn': 'এই প্রশ্নের সঠিক উত্তর দেওয়ার মতো পর্যাপ্ত যাচাইকৃত তথ্য MSMARCO-XI ডেটাসেটে নেই।',
            'te': 'ఈ ప్రశ్నకు ఖచ్చితమైన సమాధానం ఇవ్వడానికి MSMARCO-XI డేటాసెట్‌లో తగిన సమాచారం లేదు.',
            'ta': 'இந்தக் கேள்விக்கு துல்லியமாக பதிலளிக்க MSMARCO-XI தரவுத்தொகுப்பில் போதிய தகவல் இல்லை.',
            'mr': 'या प्रश्नाचे अचूक उत्तर देण्यासाठी MSMARCO-XI डेटाबेसमध्ये पुरेशी माहिती उपलब्ध नाही.'
        }
        return refusals.get(language, refusals['en'])
