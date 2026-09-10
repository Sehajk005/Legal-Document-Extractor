# ⚖️ AI-Powered Legal Aid for Common Citizens

**Upload a legal document. See each clause explained in plain English with its risks flagged, then ask questions that get answered only from that document.**

![App demo](./assets/demo.gif)

**[Watch the 3-minute demo](https://youtu.be/dzDFZkL8zMQ)** · Built for the Cosdata Hackathon 2025

[Features](#what-it-does) · [Architecture](#how-it-works) · [Gatekeeper](#the-document-gatekeeper) · [Problems solved](#problems-i-ran-into) · [Setup](#getting-started) · [Roadmap](#roadmap)

> [!IMPORTANT]
> This project explains what a legal document says. It does not give legal advice and is not a substitute for a lawyer.

## Why I built this

Legal documents are written in language most people can't comfortably read, and paying a lawyer to go through every agreement isn't realistic for most citizens. So people sign things they only half understand.

I wanted a tool where anyone could upload a document, find out what each clause actually means, see what could go wrong before signing, and ask follow-up questions in plain language, without the tool pretending to be their lawyer.

## What it does

| Feature | Details |
|---|---|
| Document gatekeeper | A three-stage check (regex, zero-shot classifier, small local LLM) rejects files that aren't legal documents before any analysis runs. |
| Hybrid parsing | Digital PDFs are read directly with PyMuPDF. Scanned PDFs are detected automatically and sent through OpenCV preprocessing and Tesseract OCR. |
| Entity extraction | Gemini pulls out individuals, companies, organizations, dates, addresses, emails and phone numbers. |
| Clause analysis | Each clause gets a title, a type (Termination, Payment, Liability, Confidentiality, Governing Law, Force Majeure, General or Other), its original text, a plain-English summary, and its potential risks, shown in red. |
| Document Q&A | A RAG chatbot answers from the most relevant chunks of the user's own document. |
| Responsible answers | Advice questions get the facts plus a clear "I can't advise you" note. Malicious requests are refused. If the answer isn't in the document, the bot says so. |
| Feedback | Every answer has 👍 / 👎 buttons and a comment box, and feedback is logged to Google Sheets. |

## How it works

```mermaid
flowchart TD
    subgraph Parsing["Upload and parsing"]
        U["PDF upload"] --> P["PyMuPDF text extraction"]
        P -- "more than 100 characters" --> G["3-stage document gatekeeper"]
        P -- "100 characters or fewer" --> IMG["Render pages to images"]
        IMG --> PRE["OpenCV grayscale and Otsu threshold"]
        PRE --> OCR["Tesseract OCR"]
        OCR --> G
    end

    G -- "not legal" --> X["Rejected"]

    subgraph Analysis["Document analysis"]
        G -- "legal" --> LLM["Gemini Flash: entities and clauses as JSON"]
        LLM --> UI["Streamlit report: entities, plain-English clauses, risks in red"]
    end

    subgraph Retrieval["Indexing and Q&A"]
        G -- "legal" --> CH["800-character chunks with 100 overlap"]
        CH --> EMB["all-MiniLM-L6-v2 embeddings"]
        EMB --> DB[("Cosdata global collection, vector IDs scoped to session and document")]
        Q["User question"] --> S["Dense search, top 50"]
        DB --> S
        S --> F["Keep only this session's document chunks, up to 5"]
        F --> A["Gemini Flash answers under responsible-answer rules"]
        A --> FB["Answer with feedback logged to Google Sheets"]
    end
