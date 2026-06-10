# Cog Whisper Diarization

Audio transcribing + diarization pipeline.

## AI Models used

- Transcription: Faster Whisper Large v3 Turbo (`faster-whisper==1.2.1`)
- Diarization: `pyannote/speaker-diarization-community-1` (`pyannote.audio==4.0.4`)

## Usage

- Used at [Audiogest](https://audiogest.app)
- Or try at [Replicate](https://replicate.com/thomasmol/whisper-diarization)
- Or use a similar version with better diarization at [pyannoteAI](https://www.pyannote.ai/)
- Or build locally with [Cog](https://cog.run) and deploy on [Replicate](https://replicate.com/)

## Deploy

- Make sure you have [cog](https://cog.run) installed
- Accept [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1) user conditions
- Create a Hugging Face read token at [hf.co/settings/tokens](https://hf.co/settings/tokens) and write it to a local secret file: `.hf_token`
- Then build:

```sh
cog build -t whisper-diarization:latest --secret id=HF_TOKEN,src=.hf_token
```

- Run:

```sh
cog run -i file=@input.wav
```

- Push to Replicate:

```sh
cog push r8.im/<username>/<name>
```

- Please follow instructions on [cog.run](https://cog.run) if you run into issues

### Input

- `file_string: str`: Either provide a Base64 encoded audio file.
- `file_url: str`: Or provide a direct audio file URL.
- `file: Path`: Or provide an audio file.
- `num_speakers: int`: Number of speakers. Leave empty to autodetect. Must be between 1 and 50.
- `translate: bool`: Translate the speech into English.
- `language: str`: Language of the spoken words as a language code like 'en'. Leave empty to auto detect language.
- `prompt: str`: Vocabulary: provide names, acronyms, and loanwords in a list. Use punctuation for best accuracy.

### Output

- `segments: List[Dict]`: List of segments with speaker, start and end time.
  - Includes `avg_logprob` for each segment and `probability` for each word level segment.
- `num_speakers: int`: Number of speakers (detected, unless specified in input).
- `language: str`: Language of the spoken words as a language code like 'en' (detected, unless specified in input).

## Thanks to

- [pyannote](https://github.com/pyannote/pyannote-audio) - Speaker diarization model
- [whisper](https://github.com/openai/whisper) - Speech recognition model
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Reimplementation of Whisper model for faster inference
- [cog](https://github.com/replicate/cog) - ML containerization framework
