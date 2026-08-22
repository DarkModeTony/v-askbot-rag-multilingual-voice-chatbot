/**
 * v-askbot Frontend Controller
 * Hacker House Goa 2026 Edition
 * Integrates Sarvam AI Translation Pipeline + Groq RAG Engine + Web Speech API
 */

// Sample queries mapped by language from MSMARCO-XI dataset
const PRESET_QUERIES = {
    'hi': [
        { text: 'गोवा कहाँ स्थित है और इसकी राजधानी क्या है?', tag: 'Geography' },
        { text: 'कृत्रिम बुद्धिमत्ता में RAG क्या है?', tag: 'AI Tech' },
        { text: 'चंद्रयान-3 मिशन का प्राथमिक उद्देश्य क्या था?', tag: 'Space Science' }
    ],
    'bn': [
        { text: 'কৃত্রিম বুদ্ধিমত্তায় রিট্রিভাল-অগমেন্টেড জেনারেশন (RAG) কী?', tag: 'AI Tech' },
        { text: 'গোয়ার রাজধানী কী এবং এটি কোথায় অবস্থিত?', tag: 'Geography' },
        { text: 'ভারতের চন্দ্রযান-৩ মিশন কেন গুরুত্বপূর্ণ?', tag: 'Space Science' }
    ],
    'en': [
        { text: 'What is retrieval augmented generation in artificial intelligence?', tag: 'AI Tech' },
        { text: 'Where is Goa located and what is its capital city?', tag: 'Geography' },
        { text: 'What were the main objectives of the Chandrayaan-3 lunar mission?', tag: 'Space Science' }
    ],
    'te': [
        { text: 'ఆర్టిఫిషియల్ ఇంటెలిజెన్స్‌లో RAG అంటే ఏమిటి?', tag: 'AI Tech' },
        { text: 'గోవా రాజధాని ఏమిటి?', tag: 'Geography' }
    ],
    'ta': [
        { text: 'செயற்கை நுண்ணறிவில் RAG என்றால் என்ன?', tag: 'AI Tech' },
        { text: 'கோவாவின் தலைநகரம் எது?', tag: 'Geography' }
    ],
    'mr': [
        { text: 'आर्टिफिशियल इंटेलिजन्समध्ये RAG म्हणजे काय?', tag: 'AI Tech' },
        { text: 'गोव्याची राजधानी कोणती आहे?', tag: 'Geography' }
    ]
};

// DOM Elements
const languageSelect = document.getElementById('languageSelect');
const strategySelect = document.getElementById('strategySelect');
const presetChipsContainer = document.getElementById('presetChips');
const micButton = document.getElementById('micButton');
const micStatusLabel = document.getElementById('micStatusLabel');
const queryInput = document.getElementById('queryInput');
const submitQueryBtn = document.getElementById('submitQueryBtn');
const speakAnswerBtn = document.getElementById('speakAnswerBtn');
const answerContent = document.getElementById('answerContent');
const translationBox = document.getElementById('translationBox');
const englishQueryText = document.getElementById('englishQueryText');
const englishAnswerText = document.getElementById('englishAnswerText');
const transLatencyBadge = document.getElementById('transLatencyBadge');
const guardrailStatus = document.getElementById('guardrailStatus');
const groundedScore = document.getElementById('groundedScore');
const totalLatency = document.getElementById('totalLatency');
const citationsList = document.getElementById('citationsList');
const citationsCountBadge = document.getElementById('citationsCountBadge');
const compareStrategiesBtn = document.getElementById('compareStrategiesBtn');
const comparisonGrid = document.getElementById('comparisonGrid');

// State
let isRecording = false;
let recognition = null;
let mediaRecorder = null;
let recordingStream = null;
let recordedAudio = [];
let currentAnswerText = '';
let currentLanguage = 'hi';

// Initialize Web Speech API for Microphone input
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        micStatusLabel.textContent = 'Click Mic to record, then click again to send';
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => {
        isRecording = true;
        micButton.classList.add('recording');
        micStatusLabel.textContent = 'Listening... Speak now';
        updateTrackerStep('nodeInput', 'active');
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        queryInput.value = transcript;
        micStatusLabel.textContent = 'Captured: "' + transcript.substring(0, 30) + '..."';
        // Auto-submit voice query
        executeRAGQuery(transcript);
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        stopRecording();
        if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
            recognition = null;
            micStatusLabel.textContent = 'Speech access blocked; switching to audio recording...';
            startAudioRecording();
        } else {
            micStatusLabel.textContent = 'Mic error (' + event.error + '). Click to retry';
        }
    };

    recognition.onend = () => {
        stopRecording();
    };
}

function stopRecording() {
    isRecording = false;
    micButton.classList.remove('recording');
    if (micStatusLabel.textContent.includes('Listening')) {
        micStatusLabel.textContent = 'Click Mic to Speak in Selected Language';
    }
}

async function startAudioRecording() {
    if (!window.isSecureContext) {
        micStatusLabel.textContent = 'Open this app at http://localhost:8000 to enable the microphone';
        return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
        micStatusLabel.textContent = 'Audio recording is unavailable in this browser';
        return;
    }

    try {
        recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        recordedAudio = [];
        mediaRecorder = new MediaRecorder(recordingStream);
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) recordedAudio.push(event.data);
        };
        mediaRecorder.onstop = submitRecordedAudio;
        mediaRecorder.start();
        isRecording = true;
        micButton.classList.add('recording');
        micStatusLabel.textContent = 'Recording... click Mic again when finished';
        updateTrackerStep('nodeInput', 'active');
    } catch (error) {
        console.error('Unable to access microphone:', error);
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            micStatusLabel.textContent = 'Allow Microphone for localhost, then click Mic again';
        } else {
            micStatusLabel.textContent = `Microphone error: ${error.message}`;
        }
    }
}

async function submitRecordedAudio() {
    const blob = new Blob(recordedAudio, { type: mediaRecorder?.mimeType || 'audio/webm' });
    const formData = new FormData();
    formData.append('audio', blob, 'voice-query.webm');
    formData.append('language', languageSelect.value);
    formData.append('strategy', strategySelect.value);

    micStatusLabel.textContent = 'Transcribing and running RAG...';
    micButton.disabled = true;

    try {
        const response = await fetch('/api/voice-query', { method: 'POST', body: formData });
        if (!response.ok) throw new Error(`Voice request failed (${response.status})`);
        const data = await response.json();
        queryInput.value = data.transcript || '';
        renderQueryResult(data);
        micStatusLabel.textContent = `Captured: "${(data.transcript || '').substring(0, 30)}..."`;
    } catch (error) {
        console.error('Voice pipeline failed:', error);
        micStatusLabel.textContent = `Voice error: ${error.message}`;
    } finally {
        recordingStream?.getTracks().forEach(track => track.stop());
        recordingStream = null;
        mediaRecorder = null;
        recordedAudio = [];
        micButton.disabled = false;
        stopRecording();
    }
}

// Render Presets when Language changes
function renderPresetChips(lang) {
    presetChipsContainer.innerHTML = '';
    const samples = PRESET_QUERIES[lang] || PRESET_QUERIES['en'];

    samples.forEach(sample => {
        const chip = document.createElement('div');
        chip.className = 'preset-chip';
        chip.innerHTML = `
            <span>${sample.text}</span>
            <span class="preset-tag">${sample.tag}</span>
        `;
        chip.addEventListener('click', () => {
            queryInput.value = sample.text;
            executeRAGQuery(sample.text);
        });
        presetChipsContainer.appendChild(chip);
    });
}

// Visual Pipeline Node State Updates
function updateTrackerStep(nodeId, state) {
    const node = document.getElementById(nodeId);
    if (!node) return;
    node.classList.remove('active', 'completed');
    if (state) node.classList.add(state);
}

function resetTracker() {
    ['nodeInput', 'nodeTranslateIn', 'nodeGroq', 'nodeTranslateOut'].forEach(id => {
        updateTrackerStep(id, '');
    });
}

function renderQueryResult(data) {
    updateTrackerStep('nodeTranslateIn', 'completed');
    updateTrackerStep('nodeGroq', 'completed');
    updateTrackerStep('nodeTranslateOut', 'completed');

    currentAnswerText = data.answer || '';
    currentLanguage = data.language || languageSelect.value;
    answerContent.innerHTML = `<p>${currentAnswerText}</p>`;
    speakAnswerBtn.disabled = !currentAnswerText;

    if (data.translation && data.language !== 'en') {
        translationBox.style.display = 'block';
        englishQueryText.textContent = data.english_query || '--';
        englishAnswerText.textContent = data.english_answer || '--';
        const totalTransMs = (data.latency_breakdown.translate_in_ms || 0) + (data.latency_breakdown.translate_out_ms || 0);
        transLatencyBadge.textContent = `${totalTransMs.toFixed(1)} ms total translation`;
    } else {
        translationBox.style.display = 'none';
    }

    const guardStatus = data.guardrails?.status || 'UNKNOWN';
    guardrailStatus.textContent = guardStatus;
    guardrailStatus.className = guardStatus === 'PASSED_GROUNDED'
        ? 'status-value status-badge-clean'
        : 'status-value badge-warning';
    groundedScore.textContent = `${((data.confidence || 0) * 100).toFixed(0)}% Grounded`;
    totalLatency.textContent = `${(data.latency_breakdown?.total_e2e_ms || 0).toFixed(1)} ms`;
    renderCitations(data.citations);
    updateLatencyWaterfall(data.latency_breakdown || {});
}

// Execute the full RAG Pipeline via FastAPI
async function executeRAGQuery(queryText) {
    const query = queryText || queryInput.value.trim();
    if (!query) {
        alert('Please enter or speak a question first.');
        return;
    }

    const language = languageSelect.value;
    const strategy = strategySelect.value;

    // UI Loading state
    submitQueryBtn.disabled = true;
    submitQueryBtn.innerHTML = '<span>Processing Pipeline...</span>';
    answerContent.innerHTML = '<p class="placeholder-text">Executing Sarvam Translation → Groq Vector RAG → Translation...</p>';
    speakAnswerBtn.disabled = true;

    resetTracker();
    updateTrackerStep('nodeInput', 'completed');
    updateTrackerStep('nodeTranslateIn', 'active');

    try {
        const response = await fetch('/api/text-query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                language: language,
                strategy: strategy
            })
        });

        if (!response.ok) throw new Error(`Query failed (${response.status})`);
        const data = await response.json();
        renderQueryResult(data);

    } catch (err) {
        console.error('Pipeline execution failed:', err);
        answerContent.innerHTML = `<p style="color: var(--color-danger)">Execution error: ${err.message}</p>`;
    } finally {
        submitQueryBtn.disabled = false;
        submitQueryBtn.innerHTML = `
            <span>Execute Sub-200ms RAG</span>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <line x1="5" y1="12" x2="19" y2="12"></line>
                <polyline points="12 5 19 12 12 19"></polyline>
            </svg>
        `;
    }
}

// Render Retrieved Citations
function renderCitations(citations) {
    if (!citations || citations.length === 0) {
        citationsList.innerHTML = '<div class="empty-citations">No verified passages retrieved for this query.</div>';
        citationsCountBadge.textContent = '0 Sources';
        return;
    }

    citationsCountBadge.textContent = `${citations.length} Sources Verified`;
    citationsList.innerHTML = '';

    citations.forEach(c => {
        const item = document.createElement('div');
        item.className = 'citation-item';
        item.innerHTML = `
            <div class="citation-header">
                <span class="citation-domain">${c.source_domain} • Chunk ${c.chunk_id}</span>
                <span class="citation-score">Relevance: ${(c.relevance_score * 100).toFixed(1)}%</span>
            </div>
            <div class="citation-excerpt">${c.excerpt}</div>
        `;
        citationsList.appendChild(item);
    });
}

// Update Waterfall Telemetry Bars
function updateLatencyWaterfall(breakdown) {
    const total = breakdown.total_e2e_ms || 1;

    document.getElementById('valTransIn').textContent = `${(breakdown.translate_in_ms || 0).toFixed(1)}ms`;
    document.getElementById('barTransIn').style.width = `${Math.min(100, ((breakdown.translate_in_ms || 0) / total) * 100)}%`;

    document.getElementById('valVector').textContent = `${(breakdown.retrieval_ms || 0).toFixed(1)}ms`;
    document.getElementById('barVector').style.width = `${Math.min(100, ((breakdown.retrieval_ms || 0) / total) * 100)}%`;

    document.getElementById('valGroq').textContent = `${(breakdown.llm_total_ms || 0).toFixed(1)}ms`;
    document.getElementById('barGroq').style.width = `${Math.min(100, ((breakdown.llm_total_ms || 0) / total) * 100)}%`;

    document.getElementById('valTransOut').textContent = `${(breakdown.translate_out_ms || 0).toFixed(1)}ms`;
    document.getElementById('barTransOut').style.width = `${Math.min(100, ((breakdown.translate_out_ms || 0) / total) * 100)}%`;

    document.getElementById('valGuard').textContent = `${(breakdown.guardrails_ms || 0).toFixed(1)}ms`;
    document.getElementById('barGuard').style.width = `${Math.min(100, ((breakdown.guardrails_ms || 0) / total) * 100)}%`;
}

// Native Browser Speech Synthesis (TTS)
function speakAnswer() {
    if (!currentAnswerText) return;
    if (!window.speechSynthesis) {
        speakAnswerBtn.textContent = 'TTS unavailable';
        return;
    }

    window.speechSynthesis.cancel(); // Stop any previous speech

    const utterance = new SpeechSynthesisUtterance(currentAnswerText);
    const langMap = {
        'hi': 'hi-IN',
        'bn': 'bn-IN',
        'en': 'en-US',
        'te': 'te-IN',
        'ta': 'ta-IN',
        'mr': 'mr-IN',
        'gu': 'gu-IN'
    };
    utterance.lang = langMap[currentLanguage] || 'en-US';
    utterance.rate = 0.95;

    speakAnswerBtn.textContent = '🔊 Speaking...';
    utterance.onend = () => { speakAnswerBtn.textContent = '🔊 Listen (TTS)'; };
    utterance.onerror = () => { speakAnswerBtn.textContent = '🔊 Listen (TTS)'; };

    window.speechSynthesis.speak(utterance);
}

// Vast Chunking Strategy Comparison Evaluation
async function compareChunkingStrategies() {
    compareStrategiesBtn.disabled = true;
    compareStrategiesBtn.textContent = 'Analyzing 5 Strategies...';

    const sampleText = "HackerHouse Goa is an elite AI and builder residency in Goa, India. It brings together 247 handpicked creators, developers, and researchers. The residency focuses on autonomous agents, crypto convergence, and high-performance retrieval architectures on datasets like MSMARCO-XI.";

    try {
        const resp = await fetch('/api/chunking-compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: sampleText,
                domain: 'technology',
                language: languageSelect.value
            })
        });

        const data = await resp.json();
        comparisonGrid.innerHTML = '';

        for (const [stratKey, stats] of Object.entries(data.analytics)) {
            const card = document.createElement('div');
            card.className = 'strategy-stat-card';
            card.innerHTML = `
                <div class="strat-name">${stratKey.replace(/_/g, ' ')}</div>
                <div class="strat-stat-row">
                    <span class="strat-stat-label">Chunks Created:</span>
                    <span class="strat-stat-value">${stats.num_chunks}</span>
                </div>
                <div class="strat-stat-row">
                    <span class="strat-stat-label">Avg Words / Chunk:</span>
                    <span class="strat-stat-value">${stats.avg_words_per_chunk}</span>
                </div>
                <div class="strat-stat-row">
                    <span class="strat-stat-label">Std Dev (Variance):</span>
                    <span class="strat-stat-value">${stats.std_dev}</span>
                </div>
                <div class="strat-stat-row">
                    <span class="strat-stat-label">Granularity:</span>
                    <span class="strat-stat-value">${stats.granularity}</span>
                </div>
            `;
            comparisonGrid.appendChild(card);
        }
    } catch (e) {
        console.error('Failed to compare chunking strategies:', e);
    } finally {
        compareStrategiesBtn.disabled = false;
        compareStrategiesBtn.textContent = 'Analyze 5 Strategies';
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    initSpeechRecognition();
    renderPresetChips(languageSelect.value);

    languageSelect.addEventListener('change', (e) => {
        renderPresetChips(e.target.value);
        if (recognition) {
            const langMap = {
                'hi': 'hi-IN',
                'bn': 'bn-IN',
                'en': 'en-IN',
                'te': 'te-IN',
                'ta': 'ta-IN',
                'mr': 'mr-IN',
                'gu': 'gu-IN'
            };
            recognition.lang = langMap[e.target.value] || 'en-IN';
        }
    });

    micButton.addEventListener('click', () => {
        const recording = micButton.classList.contains('recording');
        if (recording) {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
            else if (recognition) recognition.stop();
        } else if (navigator.mediaDevices?.getUserMedia && window.MediaRecorder) {
            startAudioRecording();
        } else if (recognition) {
            try {
                recognition.start();
            } catch (error) {
                micStatusLabel.textContent = 'Mic is already starting; click again in a moment';
            }
        } else {
            micStatusLabel.textContent = 'Voice input is unavailable in this browser';
        }
    });

    submitQueryBtn.addEventListener('click', () => executeRAGQuery());
    speakAnswerBtn.addEventListener('click', speakAnswer);
    compareStrategiesBtn.addEventListener('click', compareChunkingStrategies);

    // Initial chunking comparison load
    compareChunkingStrategies();
});
