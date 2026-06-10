import base64
import datetime
import subprocess
import os
import requests
import time
import torch
import re
import numpy as np

from cog import BasePredictor, BaseModel, Input, Path
from faster_whisper import WhisperModel, BatchedInferencePipeline
from pyannote.audio import Pipeline
import torchaudio
from faster_whisper.vad import VadOptions
from huggingface_hub import login

# pyannote/speaker-diarization-3.1 uses wespeaker embeddings (256-d). Stamped on
# every centroid so a future diarization-model swap is detectable downstream.
VOICEPRINT_MODEL = "pyannote-speaker-diarization-3.1"


class Output(BaseModel):
    segments: list
    language: str = None
    num_speakers: int = None
    # Per-speaker voiceprint centroids for the cross-file identity layer:
    #   {SPEAKER_xx: {embedding: [float], dim, model_version, duration_s, segment_count}}
    # One mean centroid per speaker per file (pyannote return_embeddings=True).
    # Empty dict when diarization produced no usable embeddings.
    speakers: dict = None

class Predictor(BasePredictor):
    def setup(self):
        # PERFORMANCE: Enable TF32 for 4090 (Ada Lovelace) hardware acceleration
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        model_name = "large-v3-turbo"
        model = WhisperModel(
            model_name,
            device="cuda",
            compute_type="int8_float16",
        )
        self.model = BatchedInferencePipeline(model=model)
        
        hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
        if hf_token:
            login(token=hf_token)
        
        self.diarization_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        ).to(torch.device("cuda"))

    def predict(
        self,
        file_string: str = Input(description="Base64 encoded audio", default=None),
        file_url: str = Input(description="Direct audio URL", default=None),
        file: Path = Input(description="Audio file", default=None),
        num_speakers: int = Input(description="Number of speakers (1-50)", ge=1, le=50, default=None),
        translate: bool = Input(description="Translate to English", default=False),
        language: str = Input(description="Language code (e.g. 'en')", default=None),
        prompt: str = Input(description="Vocabulary/hotwords", default=None),
    ) -> Output:
        temp_wav_filename = f"temp-{time.time_ns()}.wav"
        temp_audio_filename = None

        try:
            # 1. AUDIO DOWNLOAD/PREP
            if file:
                input_path = str(file)
            elif file_url:
                response = requests.get(file_url)
                temp_audio_filename = f"temp-{time.time_ns()}.audio"
                with open(temp_audio_filename, "wb") as f:
                    f.write(response.content)
                input_path = temp_audio_filename
            elif file_string:
                audio_data = base64.b64decode(file_string.split(",")[1] if "," in file_string else file_string)
                temp_audio_filename = f"temp-{time.time_ns()}.audio"
                with open(temp_audio_filename, "wb") as f:
                    f.write(audio_data)
                input_path = temp_audio_filename
            else:
                raise ValueError("No audio input provided.")

            # FFmpeg conversion to 16k mono wav (required for Whisper/Pyannote)
            subprocess.run([
                "ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1",
                "-c:a", "pcm_s16le", temp_wav_filename, "-y"
            ], check=True, capture_output=True)

            # 2. INFERENCE
            segments, detected_speakers, detected_lang, speaker_voiceprints = self.speech_to_text(
                temp_wav_filename, num_speakers, prompt, language, translate
            )

            return Output(
                segments=segments,
                language=detected_lang,
                num_speakers=detected_speakers,
                speakers=speaker_voiceprints,
            )

        finally:
            for f in [temp_wav_filename, temp_audio_filename]:
                if f and os.path.exists(f):
                    os.remove(f)

    def speech_to_text(self, audio_path, num_speakers, prompt, language, translate):
        # TRANSCRIPTION
        # beam_size=1 is the secret sauce for the Turbo model's speed
        options = dict(
            language=language,
            beam_size=1, 
            batch_size=16,
            vad_filter=True,
            # vad_parameters=VadOptions(
            #     max_speech_duration_s=30,
            #     min_speech_duration_ms=100,
            #     speech_pad_ms=100,
            #     threshold=0.25,
            # ),
            word_timestamps=True,
            initial_prompt=prompt,
            task="translate" if translate else "transcribe",
        )
        
        t_start = time.time()
        whisper_segments, info = self.model.transcribe(audio_path, **options)
        whisper_segments = list(whisper_segments)
        t_transcribe = time.time() - t_start

        # DIARIZATION (+ per-speaker voiceprint centroids for cross-file identity)
        # return_embeddings=True makes pyannote also return one mean embedding per
        # speaker cluster, aligned (by row) with sorted diarization.labels().
        waveform, sample_rate = torchaudio.load(audio_path)
        diarization, speaker_embeddings = self.diarization_model(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
            return_embeddings=True,
        )
        t_diarize = time.time() - (t_start + t_transcribe)

        # 3. LINEAR MERGE (O(N) Complexity - No Pandas!)
        # Flatten diarization into a sorted list of speaker turns
        diar_turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diar_turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        
        unique_speakers = len(set(d["speaker"] for d in diar_turns))
        
        final_segments = []
        diar_idx = 0

        for s in whisper_segments:
            seg_start, seg_end = s.start, s.end
            
            # Find the best speaker via maximum overlap
            best_speaker = "UNKNOWN"
            max_overlap = 0
            
            # Fast-forward diarization index to current segment start
            while diar_idx < len(diar_turns) and diar_turns[diar_idx]["end"] < seg_start:
                diar_idx += 1
            
            # Check all turns that intersect with this segment
            curr_idx = diar_idx
            while curr_idx < len(diar_turns) and diar_turns[curr_idx]["start"] < seg_end:
                overlap = min(seg_end, diar_turns[curr_idx]["end"]) - max(seg_start, diar_turns[curr_idx]["start"])
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = diar_turns[curr_idx]["speaker"]
                curr_idx += 1

            # Build words list
            words_list = []
            for w in (s.words or []):
                words_list.append({
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": w.word,
                    "probability": float(w.probability),
                    "speaker": best_speaker # Segment-level is usually enough for words
                })

            final_segments.append({
                "start": float(seg_start),
                "end": float(seg_end),
                "text": s.text.strip(),
                "speaker": best_speaker,
                "avg_logprob": float(s.avg_logprob),
                "words": words_list
            })

        # 4. SMART GROUPING
        if not final_segments:
            return [], 0, info.language

        grouped = []
        current = final_segments[0]
        end_punctuation = r"[.!?]$"

        for i in range(1, len(final_segments)):
            next_seg = final_segments[i]
            gap = next_seg["start"] - current["end"]
            duration = current["end"] - current["start"]

            # Merge if same speaker, small gap, and not a full sentence yet
            if (next_seg["speaker"] == current["speaker"] and 
                gap <= 1.0 and 
                duration < 25.0 and 
                not re.search(end_punctuation, current["text"])):
                
                current["end"] = next_seg["end"]
                current["text"] += " " + next_seg["text"]
                current["words"].extend(next_seg["words"])
            else:
                grouped.append(current)
                current = next_seg
        
        grouped.append(current)
        
        # Cleanup final text strings
        for g in grouped:
            g["text"] = re.sub(r"\s+", " ", g["text"]).strip()
            g["duration"] = g["end"] - g["start"]

        # 5. VOICEPRINT CENTROIDS (one mean embedding per speaker per file)
        speaker_voiceprints = self._build_speaker_voiceprints(
            diarization, speaker_embeddings, final_segments
        )

        print(f"Transcribe: {t_transcribe:.2f}s | Diarize: {t_diarize:.2f}s | Total: {time.time()-t_start:.2f}s")
        return grouped, unique_speakers, info.language, speaker_voiceprints

    @staticmethod
    def _build_speaker_voiceprints(diarization, speaker_embeddings, final_segments):
        """Map pyannote's per-speaker mean embeddings to a serializable dict.

        pyannote (return_embeddings=True) yields an (n_speakers, dim) array whose
        rows align with sorted ``diarization.labels()``. Speakers with too little
        speech can have NaN/empty embeddings; those are skipped so we never store
        a poisoned centroid. ``duration_s``/``segment_count`` come from the
        whisper-aligned segments and act as a quality signal downstream.
        """
        voiceprints = {}
        if speaker_embeddings is None:
            return voiceprints

        labels = list(diarization.labels())
        for idx, label in enumerate(labels):
            if idx >= len(speaker_embeddings):
                continue
            emb = np.asarray(speaker_embeddings[idx], dtype=float)
            if emb.size == 0 or not np.all(np.isfinite(emb)):
                # Too little speech to embed reliably -> degrade to text-only.
                continue
            voiceprints[label] = {
                "embedding": [float(x) for x in emb.tolist()],
                "dim": int(emb.shape[-1]),
                "model_version": VOICEPRINT_MODEL,
                "duration_s": 0.0,
                "segment_count": 0,
            }

        for seg in final_segments:
            sp = seg.get("speaker")
            vp = voiceprints.get(sp)
            if vp is not None:
                vp["duration_s"] += float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
                vp["segment_count"] += 1

        return voiceprints
