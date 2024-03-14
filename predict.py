# Prediction interface for Cog ⚙️
from typing import Any, List
import base64
import datetime
import subprocess
import os
import requests
import time
import torch
import re

from cog import BasePredictor, BaseModel, Input, File, Path
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline


class Output(BaseModel):
    segments: list
    language: str = None
    num_speakers: int = None


class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        model_name = "large-v3"
        self.model = WhisperModel(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16",
        )
        self.diarization_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="YOUR HF TOKEN HERE",
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
        group_segments: bool = Input(
            description="Group segments of same speaker shorter apart than 2 seconds",
            default=True,
        ),
        transcript_output_format: str = Input(
            description="Specify the format of the transcript output: individual words with timestamps, full text of segments, or a combination of both.",
            default="both",
            choices=["words_only", "segments_only", "both"],
        ),
        num_speakers: int = Input(
            description="Number of speakers, leave empty to autodetect.",
            ge=1,
            le=50,
            default=None,
        ),
        language: str = Input(
            description="Language of the spoken words as a language code like 'en'. Leave empty to auto detect language.",
            default=None,
        ),
        prompt: str = Input(
            description="Vocabulary: provide names, acronyms and loanwords in a list. Use punctuation for best accuracy.",
            default=None,
        ),
        # word_timestamps: bool = Input(description="Return word timestamps", default=True), needs to be implemented
        offset_seconds: int = Input(
            description="Offset in seconds, used for chunked inputs", default=0, ge=0
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
                offset_seconds,
                group_segments,
                language,
                word_timestamps=True,
                transcript_output_format=transcript_output_format,
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
        offset_seconds=0,
        group_segments=True,
        language=None,
        word_timestamps=True,
        transcript_output_format="both",
    ):
        time_start = time.time()

        # Transcribe audio
        print("Starting transcribing")
        options = dict(
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=1000),
            initial_prompt=prompt,
            word_timestamps=word_timestamps,
            language=language,
        )
        segments, transcript_info = self.model.transcribe(audio_file_wav, **options)
        segments = list(segments)
        segments = [
            {
                "avg_logprob": s.avg_logprob,
                "start": float(s.start + offset_seconds),
                "end": float(s.end + offset_seconds),
                "text": s.text,
                "words": [
                    {
                        "start": float(w.start + offset_seconds),
                        "end": float(w.end + offset_seconds),
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
            f"Finished with transcribing, took {time_transcribing_end - time_start:.5} seconds"
        )

        print("Starting diarization")
        diarization = self.diarization_model(audio_file_wav, num_speakers=num_speakers)

        time_diraization_end = time.time()
        print(
            f"Finished with diarization, took {time_diraization_end - time_transcribing_end:.5} seconds"
        )

        print("Starting merging")

        # Initialize variables to keep track of the current position in both lists
        margin = 0.1  # 0.1 seconds margin

        # Initialize an empty list to hold the final segments with speaker info
        final_segments = []

        diarization_list = list(diarization.itertracks(yield_label=True))
        unique_speakers = {
            speaker for _, _, speaker in diarization.itertracks(yield_label=True)
        }
        detected_num_speakers = len(unique_speakers)

        speaker_idx = 0
        n_speakers = len(diarization_list)

        # Iterate over each segment
        for segment in segments:
            segment_start = segment["start"] + offset_seconds
            segment_end = segment["end"] + offset_seconds
            segment_text = []
            segment_words = []

            # Iterate over each word in the segment
            for word in segment["words"]:
                word_start = word["start"] + offset_seconds - margin
                word_end = word["end"] + offset_seconds + margin

                while speaker_idx < n_speakers:
                    turn, _, speaker = diarization_list[speaker_idx]

                    if turn.start <= word_end and turn.end >= word_start:
                        # Add word without modifications
                        segment_text.append(word["word"])

                        # Strip here for individual word storage
                        word["word"] = word["word"].strip()
                        segment_words.append(word)

                        if turn.end <= word_end:
                            speaker_idx += 1

                        break
                    elif turn.end < word_start:
                        speaker_idx += 1
                    else:
                        break

            if segment_text:
                combined_text = "".join(segment_text)
                cleaned_text = re.sub("  ", " ", combined_text).strip()
                new_segment = {
                    "avg_logprob": segment["avg_logprob"],
                    "start": segment_start - offset_seconds,
                    "end": segment_end - offset_seconds,
                    "speaker": speaker,
                    "text": cleaned_text,
                    "words": segment_words,
                }
                final_segments.append(new_segment)

        time_merging_end = time.time()
        print(
            f"Finished with merging, took {time_merging_end - time_diraization_end:.5} seconds"
        )

        print("Starting cleaning")
        segments = final_segments
        # Make output
        output = []  # Initialize an empty list for the output

        # Initialize the first group with the first segment
        current_group = {
            "start": str(segments[0]["start"]),
            "end": str(segments[0]["end"]),
            "speaker": segments[0]["speaker"],
            "avg_logprob": segments[0]["avg_logprob"],
        }

        if transcript_output_format in ("segments_only", "both"):
            current_group["text"] = segments[0]["text"]
        if transcript_output_format in ("words_only", "both"):
            current_group["words"] = segments[0]["words"]

        for i in range(1, len(segments)):
            # Calculate time gap between consecutive segments
            time_gap = segments[i]["start"] - segments[i - 1]["end"]

            # If the current segment's speaker is the same as the previous segment's speaker,
            # and the time gap is less than or equal to 2 seconds, group them
            if segments[i]["speaker"] == segments[i - 1]["speaker"] and time_gap <= 2 and group_segments:
                current_group["end"] = str(segments[i]["end"])
                if transcript_output_format in ("segments_only", "both"):
                    current_group["text"] += " " + segments[i]["text"]
                if transcript_output_format in ("words_only", "both"):
                    current_group.setdefault("words", []).extend(segments[i]["words"])
            else:
                # Add the current_group to the output list
                output.append(current_group)

                # Start a new group with the current segment
                current_group = {
                    "start": str(segments[i]["start"]),
                    "end": str(segments[i]["end"]),
                    "speaker": segments[i]["speaker"],
                    "avg_logprob": segments[i]["avg_logprob"],
                }
                if transcript_output_format in ("segments_only", "both"):
                    current_group["text"] = segments[i]["text"]
                if transcript_output_format in ("words_only", "both"):
                    current_group["words"] = segments[i]["words"]

        # Add the last group to the output list
        output.append(current_group)

        time_cleaning_end = time.time()
        print(
            f"Finished with cleaning, took {time_cleaning_end - time_merging_end:.5} seconds"
        )
        time_end = time.time()
        time_diff = time_end - time_start

        system_info = f"""Processing time: {time_diff:.5} seconds"""
        print(system_info)
        return output, detected_num_speakers, transcript_info.language
