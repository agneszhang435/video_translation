# Starbucks Video Localization Pipeline (English to Japanese)

An automated pipeline to transcribe English videos into text segments using **Fun-ASR-Nano** locally on CPU, translate them into natural localized Japanese narration with the **Gemini 2.5 API**, and synthesize a new video with matched Japanese audio tracks via **gTTS** and **FFmpeg**.

## Prerequisites
Make sure you have `ffmpeg` installed on your system.

## Setup
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Export your Gemini API key:
   ```bash
   export GOOGLE_API_KEY="your_gemini_api_key"
   ```

## Usage
Place your video named `test.mp4` in the project root directory and run:
```bash
python transcribe_and_translate.py
```
The program will output the localized Japanese audio edition of your video as `test_japanese_edition.mp4`.
