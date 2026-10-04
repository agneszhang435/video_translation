import os
import argparse
import subprocess
import soundfile as sf
import numpy as np
import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from google import genai
from gtts import gTTS

# Speed factor multipliers map for different target languages to ensure natural fits
LANGUAGE_SPEED_FACTORS = {
    'de': 1.25,  # German text expands significantly; we speed it up
    'fr': 1.15,  # French needs a slight speed boost
    'es': 1.10,  # Spanish
    'ja': 1.00,  # Japanese matches well at 1.0x
    'en': 1.00
}

def extract_audio(video_path, wav_path):
    print("Extracting audio from video...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s16le", wav_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def transcribe_audio(wav_path):
    print("Loading audio file...")
    waveform, sample_rate = sf.read(wav_path, dtype="float32")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    print("Loading Fun-ASR model...")
    model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, trust_remote_code=False, dtype=torch.float32
    ).eval()

    segment_duration = 30
    segment_length = segment_duration * sample_rate
    total_samples = len(waveform)
    transcriptions = []

    print("Transcribing in 30s segments...")
    for i in range(0, total_samples, segment_length):
        chunk = waveform[i : i + segment_length]
        if len(chunk) < 0.5 * sample_rate:
            continue
        inputs = processor.apply_transcription_request(
            audio=chunk, language="en",
            processor_kwargs={"return_tensors": "pt", "audio_kwargs": {"sampling_rate": 16000}},
        )
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=128, do_sample=True,
                temperature=0.3, top_p=0.95, repetition_penalty=1.2
            )
        new_tokens = generated[:, inputs.input_ids.shape[1]:]
        text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        if text.strip() != "/sil":
            transcriptions.append(text.strip())

    return transcriptions

def translate_segments(segments, target_lang, api_key):
    print(f"Translating segments to target language code '{target_lang}' via Gemini API...")
    client = genai.Client(api_key=api_key)
    translated = []
    for idx, text in enumerate(segments):
        prompt = f"Translate the following English advertisement segment into a natural, professional voiceover narration in language code '{target_lang}': \"{text}\"."
        response = client.models.generate_content(model='gemini-2.5-flash-lite', contents=prompt)
        translated.append(response.text.strip())
    return translated

def create_localized_video(video_path, translated_texts, target_lang, output_video_path):
    print(f"Synthesizing audio for '{target_lang}'...")
    speed_factor = LANGUAGE_SPEED_FACTORS.get(target_lang, 1.0)
    
    for idx, text in enumerate(translated_texts):
        # Save raw TTS segment
        temp_segment = f"temp_segment_{idx}.mp3"
        final_segment = f"segment_{idx}.mp3"
        
        tts = gTTS(text=text, lang=target_lang)
        tts.save(temp_segment)
        
        # Adjust speed dynamically if speed_factor is not 1.0
        if speed_factor != 1.0:
            # Use ffmpeg's atempo audio filter to adjust speech speed without changing pitch
            subprocess.run([
                "ffmpeg", "-y", "-i", temp_segment,
                "-filter:a", f"atempo={speed_factor}",
                final_segment
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(temp_segment)
        else:
            os.rename(temp_segment, final_segment)

    with open("input_audio_list.txt", "w") as f:
        for idx in range(len(translated_texts)):
            f.write(f"file 'segment_{idx}.mp3'\n")

    # Concat Audio
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", "input_audio_list.txt", "-c", "copy", "localized_full_audio.mp3"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Clean up segments
    for idx in range(len(translated_texts)):
        os.remove(f"segment_{idx}.mp3")

    # Mux Video and New Sound
    print("Combining video and localized audio track...")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-i", "localized_full_audio.mp3",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest",
        output_video_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Finished! Output saved at: {output_video_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Translate and localize video narration.")
    parser.add_argument("--video", default="test.mp4", help="Path to source video file.")
    parser.add_argument("--language", "-l", default="ja", help="Target language code (e.g. ja, de, fr, es).")
    parser.add_argument("--output", "-o", default="test_localized_edition.mp4", help="Path to output video file.")
    args = parser.parse_args()

    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("Please set GOOGLE_API_KEY environment variable.")
    else:
        extract_audio(args.video, "test_extracted.wav")
        eng_segments = transcribe_audio("test_extracted.wav")
        translated_segments = translate_segments(eng_segments, args.language, api_key)
        create_localized_video(args.video, translated_segments, args.language, args.output)
