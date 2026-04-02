document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    
    // Part 1: Auto-Detect & Transliterate
    const part1InputText = document.getElementById('part1-input-text');
    const detectedScriptLabel = document.getElementById('detected-script-label');
    const part1KaithiOutput = document.getElementById('part1-kaithi-output');
    const part1Status = document.getElementById('part1-status');
    const copyToProcessorBtn = document.getElementById('copy-to-processor-btn');

    // Part 2: Kaithi Processor (Bhashini API)
    const kaithiInputText = document.getElementById('kaithi-input-text');
    const bhashiniTargetLang = document.getElementById('bhashini-target-lang');
    const bhashiniProcessBtn = document.getElementById('bhashini-process-btn');
    const bhashiniStatus = document.getElementById('bhashini-status');
    const bhashiniTransliteratedText = document.getElementById('bhashini-transliterated-text');
    const finalTranslatedText = document.getElementById('final-translated-text');

    // Constants
    const BHASHINI_AUTH_KEY = 'qxQir6GBzhSLwk1grWI0WN9fhg0DIRnL4u8IESCEfKwVJG5MLTchujilA0mpydFK';
    const BHASHINI_ENDPOINT = 'https://dhruva-api.bhashini.gov.in/services/inference/pipeline';

    let debounceTimer;

    // --- PART 1: Auto-Detect & Kaithi Transliteration ---
    
    part1InputText.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        const text = part1InputText.value.trim();
        
        if (!text) {
            part1KaithiOutput.value = '';
            detectedScriptLabel.textContent = 'None';
            detectedScriptLabel.style.color = '#a1a1aa';
            setIndicator(part1Status, 'idle');
            return;
        }

        setIndicator(part1Status, 'loading');
        debounceTimer = setTimeout(() => {
            processPart1(text);
        }, 500);
    });

    copyToProcessorBtn.addEventListener('click', () => {
        if (part1KaithiOutput.value) {
            kaithiInputText.value = part1KaithiOutput.value;
            // Scroll to processor
            kaithiInputText.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            // Temporary flash effect to show it was copied
            kaithiInputText.style.transition = 'background-color 0.3s';
            kaithiInputText.style.backgroundColor = 'rgba(59, 130, 246, 0.2)'; // Tailwind blue-500 tint
            setTimeout(() => {
                kaithiInputText.style.backgroundColor = 'var(--bg-card)';
            }, 500);
        }
    });

    async function processPart1(text) {
        try {
            // Step 1: Auto-detect
            const detectRes = await fetch('https://www.aksharamukha.com/api/autodetect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            if (!detectRes.ok) throw new Error('Failed to auto-detect script');
            const detectedScript = await detectRes.text();
            
            if (!detectedScript || detectedScript === 'None') {
                detectedScriptLabel.textContent = 'Could not detect context';
                detectedScriptLabel.style.color = '#f87171'; // red-400
                part1KaithiOutput.value = "Aksharamukha couldn't detect the script.";
                setIndicator(part1Status, 'error');
                return;
            }

            detectedScriptLabel.textContent = detectedScript;
            detectedScriptLabel.style.color = '#60a5fa'; // blue-400

            // Step 2: Transliterate to Kaithi
            const payload = {
                "source": detectedScript,
                "target": "Kaithi",
                "text": text,
                "nativize": true,
                "postOptions": ["KaithiRetainSpace", "romanNumerals"],
                "preOptions": []
            };

            const convertRes = await fetch('https://www.aksharamukha.com/api/convert', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!convertRes.ok) throw new Error('Failed to convert to Kaithi');
            const kaithiText = await convertRes.text();
            
            part1KaithiOutput.value = kaithiText;
            setIndicator(part1Status, 'success');
        } catch (error) {
            console.error('Part 1 Error:', error);
            part1KaithiOutput.value = 'Error processing text.';
            setIndicator(part1Status, 'error');
            detectedScriptLabel.textContent = 'Error';
            detectedScriptLabel.style.color = '#f87171';
        }
    }

    // --- PART 2: Bhashini Flow ---

    bhashiniProcessBtn.addEventListener('click', async () => {
        const text = kaithiInputText.value.trim();
        if (!text) {
            alert('Please paste Kaithi text to process.');
            return;
        }

        bhashiniProcessBtn.classList.add('loading');
        bhashiniProcessBtn.disabled = true;
        setIndicator(bhashiniStatus, 'loading');
        finalTranslatedText.value = '';
        bhashiniTransliteratedText.value = '';

        try {
            // STEP 1: Transliterate Kaithi -> Devanagari
            const convertPayload = {
                "source": "Kaithi",
                "target": "Devanagari",
                "text": text,
                "nativize": true,
                "postOptions":[],
                "preOptions":[]
            };

            const convertRes = await fetch('https://www.aksharamukha.com/api/convert', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(convertPayload)
            });

            if (!convertRes.ok) throw new Error('Failed to transliterate Kaithi to Devanagari');
            const transliteratedStr = await convertRes.text();
            bhashiniTransliteratedText.value = transliteratedStr;

            // STEP 2: Translate directly using 'mai' as source language, translating Devanagari input
            finalTranslatedText.value = "Translating...";
            const tgtLang = bhashiniTargetLang.value;
            
            const translatePayload = {
                "pipelineTasks": [
                    {
                        "taskType": "translation",
                        "config": {
                            "language": {
                                "sourceLanguage": "mai", // Default assumed source language from earlier prompt configs when translating from Devanagari-based scripts mapping
                                "targetLanguage": tgtLang
                            },
                            "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4",
                            "numTranslation": "True"
                        }
                    }
                ],
                "inputData": {
                    "input": [{ "source": transliteratedStr }]
                }
            };

            const transRes = await fetch(BHASHINI_ENDPOINT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': BHASHINI_AUTH_KEY },
                body: JSON.stringify(translatePayload)
            });

            if (!transRes.ok) {
                const eTxt = await transRes.text();
                console.error("Translation fail:", eTxt);
                throw new Error('Bhashini translation failed');
            }
            const transData = await transRes.json();
            
            try {
                finalTranslatedText.value = transData.pipelineResponse[0].output[0].target;
                setIndicator(bhashiniStatus, 'success');
            } catch (e) {
                throw new Error('Could not parse Bhashini translation result. ' + JSON.stringify(transData));
            }

        } catch (error) {
            console.error('Bhashini Flow Error:', error);
            finalTranslatedText.value = error.message || 'Error executing translation flow.';
            setIndicator(bhashiniStatus, 'error');
        } finally {
            bhashiniProcessBtn.classList.remove('loading');
            bhashiniProcessBtn.disabled = false;
        }
    });

    // --- UTILITIES ---

    function setIndicator(element, state) {
        element.className = 'status-indicator';
        if (state !== 'idle') {
            element.classList.add(state);
        }
    }
});
