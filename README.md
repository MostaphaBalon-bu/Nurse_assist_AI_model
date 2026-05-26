# NurseAssist — AI Clinical Companion

A Streamlit application powered by a fine-tuned GPT-3.5-Turbo model trained on
nursing clinical scenarios. Built as a capstone project for DSC 670 at Bellevue University.

## Features
- Medication interaction and dosing checks
- SBAR handoff note generation
- Patient education scripts
- Clinical decision support
- Session audit log export

## Fine-Tuned Model
`ft:gpt-3.5-turbo-0125:personal:nurseassist-v1:DdpNvVC2`

## Setup

1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your OpenAI key:OPENAI_API_KEY=your_key_here
4. Run the app: `streamlit run app.py`

## Screenshots
![NurseAssist App](screenshots/app_screenshot.png)

## Disclaimer
Educational prototype only. Not validated for clinical use.