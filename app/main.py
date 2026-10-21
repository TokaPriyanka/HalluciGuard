
"""HalluciGuard: PDF-only, evidence-grounded multi-agent verification."""

import os
import streamlit as st
from dotenv import load_dotenv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from app.core import (
    extract_pdf_chunks,
    retrieve_evidence,
    split_into_claims,
    check_explicit_evidence,
)
from app.agents import run_agent, MODEL

load_dotenv()

st.set_page_config(
    page_title="HalluciGuard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ HalluciGuard")
st.caption("Explainable multi-agent verification for AI-generated answers")
st.write(
    "Upload trusted reference PDFs, paste an AI-generated answer, "
    "and inspect claim-level evidence and agent verdicts."
)

with st.sidebar:
    st.header("Configuration")
    st.write("Local model:", MODEL)
    st.info(
        "PDF-only mode: verdicts must be grounded in the uploaded "
        "reference documents. General knowledge is not used as evidence."
    )
    uploaded_files = st.file_uploader(
        "Trusted reference PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

answer = st.text_area(
    "AI-generated answer to verify",
    height=200,
    placeholder="Paste an answer here...",
)

verify = st.button(
    "Verify answer",
    type="primary",
    use_container_width=True,
)

if verify:
    if not answer.strip():
        st.warning("Paste an answer first.")
        st.stop()

    if not uploaded_files:
        st.warning("Upload at least one trusted PDF.")
        st.stop()

    chunks = []
    with st.spinner("Extracting PDF passages..."):
        for pdf in uploaded_files:
            try:
                chunks.extend(extract_pdf_chunks(pdf))
            except Exception as exc:
                st.error(f"Could not read {pdf.name}: {exc}")

    if not chunks:
        st.error(
            "No readable text was found. Scanned PDFs may require OCR."
        )
        st.stop()

    claims = split_into_claims(answer)
    if not claims:
        st.warning("No claims could be extracted.")
        st.stop()

    st.subheader("Verification results")
    st.caption(
        f"Checking {len(claims)} original claim(s) against "
        f"{len(chunks)} PDF passage(s)."
    )

    outcomes = []

    for index, claim in enumerate(claims, start=1):
        with st.expander(f"Claim {index}: {claim}", expanded=True):
            evidence_items = retrieve_evidence(claim, chunks, top_k=3)

            evidence_text = "\n\n".join(
                f"[Source: {item.source}, page {item.page}]\n{item.text}"
                for item in evidence_items
            )

            if evidence_items:
                st.markdown("**Retrieved PDF evidence**")
                for item in evidence_items:
                    st.markdown(
                        f"- **{item.source}, page {item.page}** "
                        f"(similarity {item.score:.2f}): {item.text}"
                    )
            else:
                st.warning("No relevant passage was retrieved.")

            # Run the three agents. Their opinions are explanatory,
            # but they do not override the deterministic evidence check.
            try:
                with st.spinner(f"Running agents for claim {index}..."):
                    trust = run_agent("Trust", claim, evidence_text)
                    skeptic = run_agent(
                        "Skeptic",
                        claim,
                        evidence_text,
                        previous_opinions=str(trust),
                    )
                    leader = run_agent(
                        "Leader",
                        claim,
                        evidence_text,
                        previous_opinions=(
                            f"Trust: {trust}\nSkeptic: {skeptic}"
                        ),
                    )
            except Exception as exc:
                st.error(
                    f"Local model call failed: {exc}. "
                    "Check that Ollama is running and the model is installed."
                )
                st.stop()

            # The evidence checker determines the final verdict.
            checked = check_explicit_evidence(claim, chunks)
            final_verdict = checked["verdict"]
            final_explanation = checked["explanation"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Trust", trust.get("verdict", "unknown"))
            c2.metric("Skeptic", skeptic.get("verdict", "unknown"))
            c3.metric("Leader", leader.get("verdict", "unknown"))

            st.markdown("**Final evidence-check result**")
            st.write(final_verdict)
            st.write(final_explanation)

            if checked.get("evidence"):
                item = checked["evidence"]
                st.caption(
                    f"Rule evidence: {item.source}, page {item.page} "
                    f"({checked.get('method', 'evidence check')})"
                )
                st.write(item.text)

            st.markdown("**Leader explanation (LLM-generated)**")
            st.write(leader.get("explanation", ""))

            if leader.get("correction"):
                st.markdown("**Agent-suggested correction**")
                st.write(leader["correction"])

            outcomes.append(final_verdict)

    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Supported", outcomes.count("supported"))
    col2.metric("Contradicted", outcomes.count("contradicted"))
    col3.metric(
        "Insufficient evidence",
        outcomes.count("insufficient_evidence"),
    )

    st.caption(
        "Research limitation: the rule-based evidence checker is "
        "conservative and does not understand every semantic contradiction. "
        "Evaluate it against labeled examples before reporting metrics."
    )
