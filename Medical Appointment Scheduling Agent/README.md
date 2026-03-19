# Appointment Scheduling Agent

A command-line medical appointment scheduling assistant built with the OpenAI API.

The agent walks the user through:
- full name
- date of birth
- insurance payer name
- chief complaint
- address collection and validation
- mock appointment selection
- final confirmation

## Features

- Step-by-step intake flow
- Date of birth validation
- Address validation with Google Maps when available
- Local address-validation fallback if Google Maps is unavailable or denies the request
- Mock appointment slot generation
- Appointment confirmation flow with yes/no confirmation

## Requirements

- Python 3.10+
- An OpenAI API key

## Setup

1. Clone this repository and move into the project directory.
2. Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_MAPS_API_KEY=your-google-maps-api-key-here
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
python chat_agent.py
```

## Notes

- `GOOGLE_MAPS_API_KEY` is optional.
- If a Google Maps key is provided, the app tries Google Geocoding first for address validation.
- If Google Maps is unavailable, misconfigured, or returns `REQUEST_DENIED`, the app falls back to built-in local validation.

## Example Address Formats

The local parser accepts either of these:

```text
123 Main St, Seattle, WA, 98101
123 Main St, Seattle, WA 98101
```
