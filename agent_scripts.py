from google.colab import userdata

# Check credentials dynamically
api_key = None
try:
    api_key = userdata.get('GOOGLE_API_KEY')
except Exception:
    pass

# Initialize autonomous actors
transcription_actor = TranscriptionAgent()
translation_actor = TranslationAgent(api_key=api_key)
production_actor = ProductionAgent()

# Launch Coordinator
orchestrator = LocalizationOrchestrator(transcription_actor, translation_actor, production_actor)

# Localize video to Japanese ('ja')
orchestrator.run_pipeline(
    video_path="/content/test.mp4",
    target_lang="ja",
    output_path="/content/agent_japanese_edition.mp4"
)