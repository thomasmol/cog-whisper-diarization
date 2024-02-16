# Cog Whisper Diarization

Audio transcribing + diarization pipeline.

## Models used

- Whisper Large v3 (CTranslate 2 version `faster-whisper`)
- Pyannote audio 3.1.1

## Usage

- Used at [Audiogest](https://audiogest.app)
- Or try at [Replicate](https://replicate.com/thomasmol/whisper-diarization)
- Or deploy yourself at [Replicate](https://replicate.com/) (Make sure to add your own HuggingFace API key and accept the terms of use of the pyannote models used)

### Input

- `file_string: str`: Either provide a Base64 encoded audio file.
- `file_url: str`: Or provide a direct audio file URL.
- `file: Path`: Or provide an audio file.
- `group_segments: bool`: Group segments of the same speaker shorter than 2 seconds apart. Default is `True`.
- `num_speakers: int`: Number of speakers. Leave empty to autodetect. Must be between 1 and 50.
- `language: str`: Language of the spoken words as a language code like 'en'. Leave empty to auto detect language.
- `prompt: str`: Vocabulary: provide names, acronyms, and loanwords in a list. Use punctuation for best accuracy.
- `offset_seconds: int`: Offset in seconds, used for chunked inputs. Default is 0.
- `transcript_output_format: str`: Specify the format of the transcript output: individual words with timestamps, full text of segments, or a combination of both.
  - Default is `both`.
  - Options are `words_only`, `segments_only`, `both`,

### Output

- `segments: List[Dict]`: List of segments with speaker, start and end time.
  - Includes `avg_logprob` for each segment and `probability` for each word level segment.
- `num_speakers: int`: Number of speakers (detected, unless specified in input).
- `language: str`: Language of the spoken words as a language code like 'en' (detected, unless specified in input).

## Thanks to

- [pyannote](https://github.com/pyannote/pyannote-audio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [whisper](https://github.com/openai/whisper)
