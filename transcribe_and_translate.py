import os
import subprocess
import soundfile as sf
import numpy as np
import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from google import genai
from gtts import gTTS

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

def translate_segments(segments, api_key):
    print("Translating segments via Gemini API...")
    client = genai.Client(api_key=api_key)
    translated = []
    for idx, text in enumerate(segments):
        prompt = f"Translate the following English advertisement segment into natural Japanese voiceover narration: \"{text}\"."
        response = client.models.generate_content(model='gemini-2.5-flash-lite', contents=prompt)
        translated.append(response.text.strip())
    return translated

def create_japanese_video(video_path, translated_texts, output_video_path):
    print("Synthesizing Japanese audio...")
    for idx, text in enumerate(translated_texts):
        tts = gTTS(text=text, lang='ja')
        tts.save(f"segment_{idx}.mp3")
        
    with open("input_audio_list.txt", "w") as f:
        for idx in range(len(translated_texts)):
            f.write(f"file 'segment_{idx}.mp3'\n")
            
    # Concat Audio
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
        "-i", "input_audio_list.txt", "-c", "copy", "japanese_full_audio.mp3"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Mux
    print("Combining video and Japanese audio track...")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-i", "japanese_full_audio.mp3",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest",
        output_video_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Finished! Output saved at: {output_video_path}")

if __name__ == '__main__':
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("Please set GOOGLE_API_KEY environment variable.")
    else:
        extract_audio("test.mp4", "test_extracted.wav")
        eng_segments = transcribe_audio("test_extracted.wav")
        jp_segments = translate_segments(eng_segments, api_key)
        create_japanese_video("test.mp4", jp_segments, "test_japanese_edition.mp4")
