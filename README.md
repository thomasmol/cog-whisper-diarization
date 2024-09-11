# Cog Whisper Diarization

Audio transcribing + diarization pipeline.

## AI/ML Models used

- Whisper Large v3 (CTranslate 2 version `faster-whisper==1.0.3`)
- Pyannote audio 3.3.1

## Usage

- Used at [Audiogest](https://audiogest.app) and [Spectropic](https://spectropic.ai)
- Or try at [Replicate](https://replicate.com/thomasmol/whisper-diarization)
- Or deploy yourself on [Replicate](https://replicate.com/) or any machine with a GPU 

## Deploy
- Make sure you have [cog](https://cog.run) installed
- Accept [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0) user conditions
- Accept [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) user conditions
- Create HuggingFace token at [hf.co/settings/tokens](https://hf.co/settings/tokens).
- Insert your own HuggingFace token in `predict.py` in the `setup` function
  - (Be careful not to commit this token!)
- Run `cog build`
- Run `cog predict -i input.wav`
  - Or push to Replicate with `cog push r8.im/<username>/<name>`
- Please follow instructions on [cog.run](https://cog.run) if you run into issues

### Input

- `file_string: str`: Either provide a Base64 encoded audio file.
- `file_url: str`: Or provide a direct audio file URL.
- `file: Path`: Or provide an audio file.
- `group_segments: bool`: Group segments of the same speaker shorter than 2 seconds apart. Default is `True`.
- `num_speakers: int`: Number of speakers. Leave empty to autodetect. Must be between 1 and 50.
- `translate: bool`: Translate the speech into English.
- `language: str`: Language of the spoken words as a language code like 'en'. Leave empty to auto detect language.
- `prompt: str`: Vocabulary: provide names, acronyms, and loanwords in a list. Use punctuation for best accuracy. Also now used as 'hotwords' paramater in transcribing,
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

- [pyannote](https://github.com/pyannote/pyannote-audio) - Speaker diarization model
- [whisper](https://github.com/openai/whisper) - Speech recognition model
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Reimplementation of Whisper model for faster inference
- [cog](https://github.com/replicate/cog) - ML containerization framework