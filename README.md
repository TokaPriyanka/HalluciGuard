# HalluciGuard — starter MVP

A document-grounded, claim-level hallucination verification prototype inspired by *Towards Detecting LLMs Hallucination via Markov Chain-based Multi-agent Debate Framework*.

## Current MVP
- Streamlit UI
- Upload trusted PDFs
- Page-aware PDF text extraction and TF-IDF evidence retrieval
- Claim extraction with an LLM (or a basic sentence-splitting fallback)
- Trust, Skeptic, and Leader agent roles
- Bounded one-pass debate in this first version
- Verdicts: `supported`, `contradicted`, `insufficient_evidence`

## Setup (Windows / Command Prompt)

```bat
cd path\to\halluciguard
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
copy .env.example .env
notepad .env
```

Put your own API key in `.env`, save, then run:

```bat
streamlit run app\main.py
```

## Safety and research notes
- Never commit `.env` or share your API key.
- Use trusted documents and inspect citations; retrieval similarity is not proof of truth.
- This prototype does not yet implement the paper's full Markov-chain debate transitions or a formal benchmark. Those are next steps.
- Do not report accuracy/F1 until the system has been tested against a labeled dataset.

## Next milestones
1. Add a reproducible test set with human-verified labels.
2. Implement explicit debate states and a maximum-round stop condition from the paper.
3. Add baseline vs RAG vs multi-agent evaluation.
4. Improve citations, correction validation, logging, and UI polish.
