# Prediction interface for Cog ⚙️
import base64
import datetime
import subprocess
import os
import requests
import time
import torch
import re
import pandas as pd
import numpy as np

from cog import BasePredictor, BaseModel, Input, Path
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torchaudio
from faster_whisper.vad import VadOptions


class Output(BaseModel):
    segments: list
    language: str = None
    num_speakers: int = None


class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        model_name = "large-v3-turbo"
        self.model = WhisperModel(
            model_name,
            device="cuda",
            compute_type="float16",
        )
        self.diarization_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="YOUR HF TOKEN",
        ).to(torch.device("cuda"))

    def predict(
        self,
        file_string: str = Input(
            description="Either provide: Base64 encoded audio file,", default=None
        ),
        file_url: str = Input(
            description="Or provide: A direct audio file URL", default=None
        ),
        file: Path = Input(description="Or an audio file", default=None),
        num_speakers: int = Input(
            description="Number of speakers, leave empty to autodetect.",
            ge=1,
            le=50,
            default=None,
        ),
        translate: bool = Input(
            description="Translate the speech into English.",
            default=False,
        ),
        language: str = Input(
            description="Language of the spoken words as a language code like 'en'. Leave empty to auto detect language.",
            default=None,
        ),
        prompt: str = Input(
            description="Vocabulary: provide names, acronyms and loanwords in a list. Use punctuation for best accuracy.",
            default=None,
        ),
    ) -> Output:
        """Run a single prediction on the model"""
        # Check if either filestring, filepath or file is provided, but only 1 of them
        """ if sum([file_string is not None, file_url is not None, file is not None]) != 1:
            raise RuntimeError("Provide either file_string, file or file_url") """

        try:
            # Generate a temporary filename
            temp_wav_filename = f"temp-{time.time_ns()}.wav"

            if file is not None:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        file,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        temp_wav_filename,
                    ]
                )

            elif file_url is not None:
                response = requests.get(file_url)
                temp_audio_filename = f"temp-{time.time_ns()}.audio"
                with open(temp_audio_filename, "wb") as file:
                    file.write(response.content)

                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        temp_audio_filename,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        temp_wav_filename,
                    ]
                )

                if os.path.exists(temp_audio_filename):
                    os.remove(temp_audio_filename)
            elif file_string is not None:
                audio_data = base64.b64decode(
                    file_string.split(",")[1] if "," in file_string else file_string
                )
                temp_audio_filename = f"temp-{time.time_ns()}.audio"
                with open(temp_audio_filename, "wb") as f:
                    f.write(audio_data)

                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        temp_audio_filename,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        temp_wav_filename,
                    ]
                )

                if os.path.exists(temp_audio_filename):
                    os.remove(temp_audio_filename)

            segments, detected_num_speakers, detected_language = self.speech_to_text(
                temp_wav_filename,
                num_speakers,
                prompt,
                language,
                translate=translate,
            )

            print(f"done with inference")
            # Return the results as a JSON object
            return Output(
                segments=segments,
                language=detected_language,
                num_speakers=detected_num_speakers,
            )

        except Exception as e:
            raise RuntimeError("Error Running inference with local model", e)

        finally:
            # Clean up
            if os.path.exists(temp_wav_filename):
                os.remove(temp_wav_filename)

    def convert_time(self, secs, offset_seconds=0):
        return datetime.timedelta(seconds=(round(secs) + offset_seconds))

    def speech_to_text(
        self,
        audio_file_wav,
        num_speakers=None,
        prompt="",
        language=None,
        translate=False,
    ):
        time_start = time.time()

        # Transcribe audio
        print("Starting transcribing")
        options = dict(
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=VadOptions(
                max_speech_duration_s=self.model.feature_extractor.chunk_length,
                min_speech_duration_ms=100,
                speech_pad_ms=100,
                threshold=0.25,
                neg_threshold=0.2,
            ),
            word_timestamps=True,
            initial_prompt=prompt,
            language_detection_segments=1,
            task="translate" if translate else "transcribe",
        )
        segments, transcript_info = self.model.transcribe(audio_file_wav, **options)
        segments = list(segments)
        segments = [
            {
                "avg_logprob": s.avg_logprob,
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text,
                "words": [
                    {
                        "start": float(w.start),
                        "end": float(w.end),
                        "word": w.word,
                        "probability": w.probability,
                    }
                    for w in s.words
                ],
            }
            for s in segments
        ]

        time_transcribing_end = time.time()
        print(
            f"Finished with transcribing, took {time_transcribing_end - time_start:.5} seconds, {len(segments)} segments"
        )

        print("Starting diarization")
        waveform, sample_rate = torchaudio.load(audio_file_wav)
        diarization = self.diarization_model(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
        )

        time_diraization_end = time.time()
        print(
            f"Finished with diarization, took {time_diraization_end - time_transcribing_end:.5} seconds"
        )

        print("Starting merging")

        # Convert diarization list to DataFrame
        diarize_segments = []
        diarization_list = list(diarization.itertracks(yield_label=True))

        for turn, _, speaker in diarization_list:
            diarize_segments.append(
                {"start": turn.start, "end": turn.end, "speaker": speaker}
            )
        diarize_df = pd.DataFrame(diarize_segments)
        unique_speakers = {speaker for _, _, speaker in diarization_list}
        detected_num_speakers = len(unique_speakers)

        # Process each segment and its words
        final_segments = []
        for segment in segments:
            # Calculate intersection for segment-level speaker assignment
            diarize_df["intersection"] = np.minimum(
                diarize_df["end"], segment["end"]
            ) - np.maximum(diarize_df["start"], segment["start"])
            diarize_df["union"] = np.maximum(
                diarize_df["end"], segment["end"]
            ) - np.minimum(diarize_df["start"], segment["start"])

            # Get speaker with maximum intersection
            dia_tmp = diarize_df[diarize_df["intersection"] > 0]
            if len(dia_tmp) > 0:
                speaker = (
                    dia_tmp.groupby("speaker")["intersection"]
                    .sum()
                    .sort_values(ascending=False)
                    .index[0]
                )
            else:
                speaker = "UNKNOWN"

            # Process words if they exist
            words_with_speakers = []
            for word in segment["words"]:
                # Calculate intersection for word-level speaker assignment
                diarize_df["intersection"] = np.minimum(
                    diarize_df["end"], word["end"]
                ) - np.maximum(diarize_df["start"], word["start"])
                diarize_df["union"] = np.maximum(
                    diarize_df["end"], word["end"]
                ) - np.minimum(diarize_df["start"], word["start"])

                # Get speaker with maximum intersection
                dia_tmp = diarize_df[diarize_df["intersection"] > 0]
                if len(dia_tmp) > 0:
                    word_speaker = (
                        dia_tmp.groupby("speaker")["intersection"]
                        .sum()
                        .sort_values(ascending=False)
                        .index[0]
                    )
                else:
                    word_speaker = (
                        speaker  # Fall back to segment speaker if no intersection
                    )

                word["speaker"] = word_speaker
                words_with_speakers.append(word)

            # Create new segment with speaker information
            new_segment = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": speaker,
                "avg_logprob": segment["avg_logprob"],
                "words": words_with_speakers,
            }
            final_segments.append(new_segment)

        # Smart grouping of segments
        if len(final_segments) > 0:
            grouped_segments = []
            current_group = final_segments[0].copy()
            sentence_end_pattern = r"[.!?]+"

            for segment in final_segments[1:]:
                time_gap = segment["start"] - current_group["end"]
                current_duration = current_group["end"] - current_group["start"]

                # Conditions for combining segments:
                # 1. Same speaker
                # 2. Time gap is reasonable (≤ 1 second)
                # 3. Current group doesn't end with sentence-ending punctuation
                # 4. Combined duration would not exceed 30 seconds
                can_combine = (
                    segment["speaker"] == current_group["speaker"]
                    and time_gap <= 1.0
                    and current_duration < 30.0
                    and not re.search(sentence_end_pattern, current_group["text"][-1:])
                )

                if can_combine:
                    # Merge segments
                    current_group["end"] = segment["end"]
                    current_group["text"] += " " + segment["text"]
                    current_group["words"].extend(segment["words"])
                else:
                    # Start new group
                    grouped_segments.append(current_group)
                    current_group = segment.copy()

            grouped_segments.append(current_group)
            final_segments = grouped_segments

        # Final cleanup of text
        for segment in final_segments:
            # Remove extra spaces
            segment["text"] = re.sub(r"\s+", " ", segment["text"]).strip()
            # Ensure proper spacing around punctuation
            segment["text"] = re.sub(r"\s+([.,!?])", r"\1", segment["text"])
            # Calculate segment duration
            segment["duration"] = segment["end"] - segment["start"]

        time_merging_end = time.time()
        print(
            f"Finished with merging, took {time_merging_end - time_diraization_end:.5} seconds"
        )

        return final_segments, detected_num_speakers, transcript_info.language
