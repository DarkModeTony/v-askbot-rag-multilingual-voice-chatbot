"""
v-askbot FastAPI Server.
Pipeline: User input → Sarvam Translate to English → Guardrails → Vector Retrieval → Groq Answer (English) → Sarvam Translate back → Output
"""
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.config import PORT, HOST, DEBUG, STATIC_DIR, TARGET_LATENCY_MS, LATENCY_FIRST_MODE
from backend.dataset_loader import MSMARCOIndicLoader, SUPPORTED_LANGUAGES
from backend.chunking_engine import VastChunkingEngine
from backend.vector_engine import HighPerformanceVectorEngine
from backend.guardrails_engine import GuardrailsEngine
from backend.model_harness import ModelHarness
from backend.stt_service import SpeechToTextService
from backend.translation_service import TranslationService
from backend.latency_profiler import LatencyProfiler

app = FastAPI(title='v-askbot: Voice Multilingual RAG', version='3.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Initialize services
dataset_loader = MSMARCOIndicLoader()
vector_engine = HighPerformanceVectorEngine()
vector_engine.index_dataset(dataset_loader)

guardrails = GuardrailsEngine()
harness = ModelHarness()
stt_service = SpeechToTextService()
translator = TranslationService()
profiler = LatencyProfiler(target_ms=TARGET_LATENCY_MS)
chunker = VastChunkingEngine()


class TextQueryRequest(BaseModel):
    query: str
    language: str = 'en'
    strategy: str = 'recursive_semantic'


class ChunkingCompareRequest(BaseModel):
    text: str
    domain: str = 'technology'
    language: str = 'en'


@app.get('/api/health')
async def health_check():
    return {
        'status': 'healthy',
        'version': '3.0.0',
        'pipeline': 'Sarvam Translate → Groq RAG → Sarvam Translate',
        'indexed_chunks': sum(len(v) for v in vector_engine.chunks_by_strategy.values()),
        'active_strategies': list(vector_engine.chunks_by_strategy.keys()),
        'supported_languages': len(SUPPORTED_LANGUAGES)
    }


@app.get('/api/languages')
async def get_languages():
    return {'languages': SUPPORTED_LANGUAGES}


@app.get('/api/metrics')
async def get_metrics():
    return profiler.get_percentiles()


@app.post('/api/chunking-compare')
async def compare_chunking(req: ChunkingCompareRequest):
    meta = {'doc_id': 'sample_doc', 'domain': req.domain, 'lang': req.language, 'query': ''}
    return chunker.evaluate_strategies(req.text, meta)


@app.post('/api/text-query')
async def process_text_query(req: TextQueryRequest):
    """
    Full pipeline:
    1. Sarvam translates user query → English
    2. Input guardrails (on English text)
    3. Vector retrieval (on English text)
    4. Groq generates answer in English
    5. Output guardrails (on English answer)
    6. Sarvam translates English answer → user's language
    """
    t_start = time.perf_counter()
    original_query = req.query
    user_lang = req.language

    # ── Step 1: Translate user input to English via Sarvam ──
    t_translate_in_start = time.perf_counter()
    if user_lang != 'en' and not LATENCY_FIRST_MODE:
        translate_in = await translator.translate_to_english(req.query, user_lang)
        english_query = translate_in['translated_text']
    else:
        translate_in = {
            'translated_text': req.query,
            'latency_ms': 0.0,
            'skipped': True,
            'service': 'latency_first_passthrough' if user_lang != 'en' else 'passthrough'
        }
        english_query = req.query
    t_translate_in = (time.perf_counter() - t_translate_in_start) * 1000

    # ── Step 2: Input Guardrails (on English text) ──
    t_g_in_start = time.perf_counter()
    in_guard = guardrails.check_input(english_query)
    t_g_in = (time.perf_counter() - t_g_in_start) * 1000

    if not in_guard['passed']:
        total_time = (time.perf_counter() - t_start) * 1000
        refusal = guardrails.generate_refusal_message(user_lang)
        return {
            'original_query': original_query,
            'english_query': english_query,
            'query': original_query,
            'answer': refusal,
            'english_answer': guardrails.generate_refusal_message('en'),
            'translated_answer': refusal,
            'language': user_lang,
            'strategy_used': req.strategy,
            'grounded': False,
            'confidence': 0.0,
            'citations': [],
            'guardrails': {
                'input_passed': False,
                'retrieval_sufficient': False,
                'groundedness_score': 0.0,
                'hallucination_detected': False,
                'status': 'BLOCKED_BY_INPUT_GUARDRAIL',
                'reason': in_guard['reason']
            },
            'translation': {
                'input_translation': translate_in,
                'output_translation': None
            },
            'tool_calls': ['guardrail:input_shield_triggered()'],
            'latency_breakdown': {
                'translate_in_ms': round(t_translate_in, 2),
                'input_guardrail_ms': round(t_g_in, 2),
                'retrieval_ms': 0.0,
                'llm_ttft_ms': 0.0,
                'llm_total_ms': 0.0,
                'output_guardrail_ms': 0.0,
                'translate_out_ms': 0.0,
                'total_e2e_ms': round(total_time, 2)
            }
        }

    # ── Step 3: Vector Retrieval (English text) ──
    t_ret_start = time.perf_counter()
    retrieved = vector_engine.search(
        query=in_guard['clean_query'],
        strategy=req.strategy,
        top_k=4,
        lang=user_lang if LATENCY_FIRST_MODE else 'en'
    )
    t_ret = (time.perf_counter() - t_ret_start) * 1000

    ret_guard = guardrails.check_retrieval_sufficiency(retrieved)
    if not ret_guard['sufficient']:
        # Hybrid routing is automatic: local MSMARCO-XI first, Groq web search second.
        t_online_start = time.perf_counter()
        online_answer, online_tools, online_ms, online_sources = harness.execute_online_search(
            query=english_query
        )

        if online_answer:
            t_translate_out_start = time.perf_counter()
            if user_lang != 'en' and not LATENCY_FIRST_MODE:
                translate_out = await translator.translate_from_english(online_answer, user_lang)
                translated_answer = translate_out['translated_text']
            else:
                translate_out = {
                    'translated_text': online_answer,
                    'latency_ms': 0.0,
                    'skipped': True,
                    'service': 'latency_first_passthrough' if user_lang != 'en' else 'passthrough'
                }
                translated_answer = online_answer
            t_translate_out = (time.perf_counter() - t_translate_out_start) * 1000
            t_total = (time.perf_counter() - t_start) * 1000
            metrics = {
                'translate_in_ms': round(t_translate_in, 2),
                'input_guardrail_ms': round(t_g_in, 2),
                'retrieval_ms': round(t_ret, 2),
                'online_search_ms': round(online_ms, 2),
                'llm_ttft_ms': round(online_ms, 2),
                'llm_total_ms': round(online_ms, 2),
                'output_guardrail_ms': 0.0,
                'translate_out_ms': round(t_translate_out, 2),
                'guardrails_ms': round(t_g_in, 2),
                'total_e2e_ms': round(t_total, 2)
            }
            profiler.record_run(metrics)
            return {
                'original_query': original_query,
                'english_query': english_query,
                'query': original_query,
                'answer': translated_answer,
                'english_answer': online_answer,
                'translated_answer': translated_answer,
                'language': user_lang,
                'strategy_used': req.strategy,
                'source_type': 'online_search',
                'grounded': False,
                'confidence': 0.0,
                'citations': online_sources,
                'guardrails': {
                    'input_passed': True,
                    'retrieval_sufficient': False,
                    'groundedness_score': 0.0,
                    'hallucination_detected': False,
                    'status': 'ONLINE_SEARCH_RESULT_UNVERIFIED',
                    'reason': 'Local dataset context was insufficient; answer came from Groq web search.'
                },
                'translation': {
                    'input_translation': translate_in,
                    'output_translation': translate_out
                },
                'tool_calls': online_tools,
                'latency_breakdown': metrics
            }

        total_time = (time.perf_counter() - t_start) * 1000
        refusal_en = guardrails.generate_refusal_message('en')
        refusal_user = guardrails.generate_refusal_message(user_lang)
        return {
            'original_query': original_query,
            'english_query': english_query,
            'query': original_query,
            'answer': refusal_user,
            'english_answer': refusal_en,
            'translated_answer': refusal_user,
            'language': user_lang,
            'strategy_used': req.strategy,
            'source_type': 'dataset_refusal',
            'grounded': False,
            'confidence': 0.0,
            'citations': [],
            'guardrails': {
                'input_passed': True,
                'retrieval_sufficient': False,
                'groundedness_score': 0.0,
                'hallucination_detected': False,
                'status': 'TRIGGERED_RETRIEVAL_REFUSAL'
            },
            'translation': {
                'input_translation': translate_in,
                'output_translation': None
            },
            'tool_calls': ['guardrail:low_relevance_refusal()'],
            'latency_breakdown': {
                'translate_in_ms': round(t_translate_in, 2),
                'input_guardrail_ms': round(t_g_in, 2),
                'retrieval_ms': round(t_ret, 2),
                'llm_ttft_ms': 0.0,
                'llm_total_ms': 0.0,
                'output_guardrail_ms': 0.0,
                'translate_out_ms': 0.0,
                'total_e2e_ms': round(total_time, 2)
            }
        }

    # ── Step 4: Groq generates answer in English ──
    t_llm_start = time.perf_counter()
    english_answer, tool_calls, ttft = harness.execute_with_harness(
        query=in_guard['clean_query'],
        context_chunks=retrieved,
        language=user_lang if LATENCY_FIRST_MODE else 'en',
        strategy=req.strategy
    )
    t_llm_total = (time.perf_counter() - t_llm_start) * 1000

    # ── Step 5: Output Guardrails (on English answer) ──
    t_g_out_start = time.perf_counter()
    out_guard = guardrails.check_output_groundedness(english_answer, retrieved)
    t_g_out = (time.perf_counter() - t_g_out_start) * 1000

    if not out_guard['grounded']:
        english_answer = guardrails.generate_refusal_message('en')

    # ── Step 6: Translate English answer back to user's language via Sarvam ──
    t_translate_out_start = time.perf_counter()
    if user_lang != 'en' and not LATENCY_FIRST_MODE:
        translate_out = await translator.translate_from_english(english_answer, user_lang)
        translated_answer = translate_out['translated_text']
    else:
        translate_out = {
            'translated_text': english_answer,
            'latency_ms': 0.0,
            'skipped': True,
            'service': 'latency_first_passthrough' if user_lang != 'en' else 'passthrough'
        }
        translated_answer = english_answer
    t_translate_out = (time.perf_counter() - t_translate_out_start) * 1000

    t_total = (time.perf_counter() - t_start) * 1000

    citations = [
        {
            'chunk_id': c['chunk_id'],
            'source_domain': c['metadata'].get('domain', 'general'),
            'relevance_score': c['score'],
            'excerpt': c['text'][:150] + '...'
        }
        for c in retrieved[:3]
    ]

    metrics = {
        'translate_in_ms': round(t_translate_in, 2),
        'input_guardrail_ms': round(t_g_in, 2),
        'retrieval_ms': round(t_ret, 2),
        'llm_ttft_ms': round(ttft, 2),
        'llm_total_ms': round(t_llm_total, 2),
        'output_guardrail_ms': round(t_g_out, 2),
        'translate_out_ms': round(t_translate_out, 2),
        'guardrails_ms': round(t_g_in + t_g_out, 2),
        'total_e2e_ms': round(t_total, 2)
    }
    profiler.record_run(metrics)

    return {
        'original_query': original_query,
        'english_query': english_query,
        'query': original_query,
        'answer': translated_answer,
        'english_answer': english_answer,
        'translated_answer': translated_answer,
        'language': user_lang,
        'strategy_used': req.strategy,
        'source_type': 'dataset',
        'grounded': out_guard['grounded'],
        'confidence': round(out_guard['groundedness_score'], 2),
        'citations': citations,
        'guardrails': {
            'input_passed': True,
            'retrieval_sufficient': True,
            'groundedness_score': out_guard['groundedness_score'],
            'hallucination_detected': out_guard['hallucination_detected'],
            'status': out_guard['verdict']
        },
        'translation': {
            'input_translation': translate_in,
            'output_translation': translate_out
        },
        'tool_calls': tool_calls,
        'latency_breakdown': metrics
    }


@app.post('/api/voice-query')
async def process_voice_query(
    audio: UploadFile = File(...),
    language: str = Form('en'),
    strategy: str = Form('recursive_semantic')
):
    """Voice input: STT → Translation → RAG → Translation → Output"""
    t_start = time.perf_counter()
    audio_content = await audio.read()

    # Sarvam STT for transcription
    stt_res = await stt_service.transcribe_audio(
        audio_content, audio.filename or 'audio.wav', f"{language}-IN"
    )
    transcript = stt_res.get('transcript', '')

    if not transcript.strip():
        raise HTTPException(
            status_code=503,
            detail=stt_res.get('error', 'Speech transcription failed. Configure Sarvam STT and try again.')
        )

    # Pass through the text pipeline
    req = TextQueryRequest(query=transcript, language=language, strategy=strategy)
    result = await process_text_query(req)
    result['transcript'] = transcript
    result['stt_service'] = stt_res.get('service', 'Sarvam AI')
    result['latency_breakdown']['stt_ms'] = stt_res.get('latency_ms', 10.0)
    result['latency_breakdown']['total_e2e_ms'] += stt_res.get('latency_ms', 10.0)

    return result


@app.post('/api/benchmark')
async def run_benchmark():
    samples = dataset_loader.get_sample_queries('en') + dataset_loader.get_sample_queries('hi')
    runs = []

    for s in samples[:15]:
        req = TextQueryRequest(query=s['query'], language=s['language'], strategy='recursive_semantic')
        res = await process_text_query(req)
        runs.append({
            'query': s['query'],
            'lang': s['language'],
            'total_e2e_ms': res['latency_breakdown']['total_e2e_ms'],
            'retrieval_ms': res['latency_breakdown']['retrieval_ms'],
            'llm_ttft_ms': res['latency_breakdown']['llm_ttft_ms'],
            'grounded': res['grounded']
        })

    percentiles = profiler.get_percentiles()
    return {
        'percentiles': percentiles,
        'runs': runs,
        'summary': f"Benchmark completed over {len(runs)} queries. P50: {percentiles['total_e2e']['p50']}ms, P70: {percentiles['total_e2e']['p70']}ms, P100: {percentiles['total_e2e']['p100']}ms (Target: <{TARGET_LATENCY_MS}ms)"
    }


app.mount('/', StaticFiles(directory=str(STATIC_DIR), html=True), name='frontend')
