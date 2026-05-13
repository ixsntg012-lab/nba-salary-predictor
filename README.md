# AI Mock Interview System 🎤

<div align="center">

**AI-powered interview practice platform — role-specific questions, instant feedback, detailed scoring**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-000000?style=for-the-badge&logo=flask)
![Gemini](https://img.shields.io/badge/Google_Gemini-AI_Powered-4285F4?style=for-the-badge&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

</div>

---

## What It Does

1. Enter your **name**, **job role**, and **experience level**
2. Gemini AI generates **5 role-specific questions** (technical + behavioral mix)
3. Answer each question in your own words
4. AI evaluates instantly and gives:
   - **Score** (0–10)
   - **Detailed feedback** (2–3 sentences)
   - **Strengths** in your answer
   - **Areas to improve**
   - **Sample strong answer** hint
5. **Final scorecard** — overall percentage + verdict

> **Problem:** Most job seekers practice interviews without feedback. Hiring coaches are expensive. This system gives honest, instant, AI-generated feedback on every answer — 24/7, free.

---

## Features

| Feature | Description |
|---------|-------------|
| Dynamic question generation | Gemini generates fresh questions for any job role |
| Real-time answer evaluation | Each answer scored and analyzed instantly |
| Strengths + improvements | Specific, actionable feedback per answer |
| Sample answer hint | AI shows what a strong answer includes |
| Final scorecard | Overall percentage + verdict (Strong / Needs Improvement / Keep Practicing) |
| Any job role | SWE, ML Engineer, Data Scientist, PM, and more |
| Clean dark UI | Progress bar, mobile-friendly, no external CSS frameworks |

---

## How It Works

```
User Input (name + role + experience level)
        │
        ▼
Gemini API — Generate 5 role-specific questions
(technical + behavioral mix, numbered list)
        │
        ▼
User answers each question
        │
        ▼
Gemini API — Evaluate answer → structured JSON response
  {
    "score": 0-10,
    "feedback": "2-3 sentence analysis",
    "strengths": ["point 1", "point 2"],
    "improvements": ["point 1", "point 2"],
    "sample": "what a strong answer includes"
  }
        │
        ▼
Final Results Page
  total score / max score → percentage → verdict
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| AI / LLM | Google Gemini API |
| Backend | Python + Flask |
| Frontend | HTML + CSS (no external frameworks) |
| Session Management | UUID-based JSON file storage (avoids cookie size limits) |

---

## Installation

```bash
git clone https://github.com/ixsntg012-lab/AI-Mock-Interview.git
cd AI-Mock-Interview
pip install -r requirements.txt
```

Add your Gemini API key in `app.py`:
```python
GEMINI_API_KEY = "your_key_here"
```

Get a free key at: [aistudio.google.com](https://aistudio.google.com)

---

## Usage

```bash
python app.py
```

Open browser: `http://localhost:5000`

---

## Project Structure

```
AI-Mock-Interview/
│
├── templates/
│   ├── index.html        ← Home (role + experience input)
│   ├── interview.html    ← Question display
│   ├── feedback.html     ← Answer evaluation + feedback
│   └── results.html      ← Final scorecard
│
├── interview_data/       ← Session JSON files (auto-created, gitignored)
├── app.py                ← Flask application + Gemini integration
├── requirements.txt
└── README.md
```

---

## Limitations & Future Work

**Phase 1 — Enhanced Evaluation**
- STAR method scoring — check if answer follows Situation, Task, Action, Result
- Domain-specific rubrics for technical vs behavioral questions
- AI follow-up questions based on your answer

**Phase 2 — Voice & Video**
- Speech-to-text input — answer by speaking
- Filler word detection ("um", "like", "you know")
- Webcam mode for body language awareness

**Phase 3 — Personalization**
- User accounts — save history and track improvement over time
- Resume-based questions — upload resume → AI generates questions from your experience
- Company-specific modes (FAANG style, startup style)

**Phase 4 — Deployment**
- Cloud deployment on Render / Railway
- Multi-language support

---

## Author

**Swetha Kiran Veernapu**
MS Computer Science

---

## License

MIT License
