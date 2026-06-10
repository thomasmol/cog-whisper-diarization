import base64
import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path as LocalPath
from typing import Optional

import numpy as np
import pandas as pd
import requests
import torch
import torchaudio
from cog import BaseModel, BaseRunner, Input, Path
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions
from pyannote.audio import Pipeline

WHISPER_MODEL_PATH = "/models/whisper/large-v3-turbo"
DIARIZATION_MODEL_PATH = "/models/diarization/pyannote--speaker-diarization-community-1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Output(BaseModel):
    segments: list
    language: Optional[str] = None
    num_speakers: Optional[int] = None


class Runner(BaseRunner):
    def setup(self) -> None:
        logger.info("Loading Whisper model from %s", WHISPER_MODEL_PATH)
        self.model = WhisperModel(
            model_size_or_path=WHISPER_MODEL_PATH,
            device="cuda",
            compute_type="float16",
        )
        logger.info("Whisper model loaded")
        logger.info("Loading diarization model from %s", DIARIZATION_MODEL_PATH)
        self.diarization_model = Pipeline.from_pretrained(DIARIZATION_MODEL_PATH).to(
            torch.device("cuda")
        )
        logger.info("Diarization model loaded")

    def run(
        self,
        file_string: Optional[str] = Input(
            description="Either provide: Base64 encoded audio file,", default=None
        ),
        file_url: Optional[str] = Input(
            description="Or provide: A direct audio file URL", default=None
        ),
        file: Optional[Path] = Input(description="Or an audio file", default=None),
        num_speakers: Optional[int] = Input(
            description="Number of speakers, leave empty to autodetect.",
            ge=1,
            le=50,
            default=None,
        ),
        translate: bool = Input(
            description="Translate the speech into English.",
            default=False,
        ),
        language: Optional[str] = Input(
            description="Language of the spoken words as a language code like 'en'. Leave empty to auto detect language.",
            default=None,
        ),
        prompt: Optional[str] = Input(
            description="Vocabulary: provide names, acronyms and loanwords in a list. Use punctuation for best accuracy.",
            default=None,
        ),
    ) -> Output:
        inputs = [file_string is not None, file_url is not None, file is not None]
        if sum(inputs) != 1:
            raise RuntimeError("Provide exactly one of file_string, file_url, or file")

        start_time = time.time()
        with tempfile.TemporaryDirectory() as directory:
            temp_dir = LocalPath(directory)
            source_path = temp_dir / "source.audio"
            wav_path = temp_dir / "audio.wav"

            if file is not None:
                source_path = LocalPath(file)
            if file_url is not None:
                download_file(file_url, source_path)
            if file_string is not None:
                write_base64_file(file_string, source_path)

            log_audio_metadata(source_path)

            normalize_start_time = time.time()
            audio_duration = normalize_audio(source_path, wav_path)
            logger.info("Audio normalized in %.2fs", time.time() - normalize_start_time)
            logger.info("Audio duration: %.2fs", audio_duration)

            segments, detected_num_speakers, detected_language = self.speech_to_text(
                str(wav_path),
                num_speakers,
                prompt,
                language,
                translate=translate,
            )
            logger.info("Run completed in %.2fs", time.time() - start_time)
            return Output(
                segments=segments,
                language=detected_language,
                num_speakers=detected_num_speakers,
            )

    def speech_to_text(
        self,
        audio_file_wav: str,
        num_speakers: Optional[int] = None,
        prompt: Optional[str] = None,
        language: Optional[str] = None,
        translate: bool = False,
    ) -> tuple[list[dict[str, object]], int, str]:
        start_time = time.time()
        gpu_type = get_gpu_type()
        logger.info("GPU type: %s", gpu_type)
        logger.info("Starting transcription")
        options = {
            "language": language,
            "beam_size": 5,
            "vad_filter": True,
            "vad_parameters": VadOptions(
                max_speech_duration_s=self.model.feature_extractor.chunk_length,
                min_speech_duration_ms=100,
                speech_pad_ms=100,
                threshold=0.25,
                neg_threshold=0.2,
            ),
            "word_timestamps": True,
            "initial_prompt": prompt,
            "language_detection_segments": 1,
            "task": "translate" if translate else "transcribe",
        }
        segments, transcript_info = self.model.transcribe(audio_file_wav, **options)
        transcription = format_transcription_segments(list(segments))
        transcribe_end_time = time.time()
        logger.info(
            "Transcription completed in %.2fs. Detected language: %s. Segments: %s",
            transcribe_end_time - start_time,
            transcript_info.language,
            len(transcription),
        )

        logger.info("Starting diarization")
        waveform, sample_rate = torchaudio.load(audio_file_wav)
        diarization = self.diarization_model(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
        )
        diarization_list = diarization.exclusive_speaker_diarization
        diarize_end_time = time.time()
        unique_speakers = {speaker for _, speaker in diarization_list}
        logger.info(
            "Diarization completed in %.2fs. Detected speakers: %s",
            diarize_end_time - transcribe_end_time,
            len(unique_speakers),
        )

        logger.info("Starting segment reconciliation")
        final_segments = post_process_segments(transcription, diarization_list)
        logger.info(
            "Segment reconciliation completed in %.2fs. Final segments: %s",
            time.time() - diarize_end_time,
            len(final_segments),
        )
        return final_segments, len(unique_speakers), transcript_info.language


def download_file(url: str, path: LocalPath) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)


def write_base64_file(file_string: str, path: LocalPath) -> None:
    encoded = file_string.split(",", 1)[1] if "," in file_string else file_string
    path.write_bytes(base64.b64decode(encoded))


def normalize_audio(input_path: LocalPath, wav_path: LocalPath) -> float:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-map",
            "0:a:0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ],
        check=True,
    )
    return get_duration(wav_path)


def get_duration(path: LocalPath) -> float:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(output.decode().strip())


def log_audio_metadata(path: LocalPath) -> None:
    try:
        metadata = torchaudio.info(str(path))
        logger.info(
            "Audio metadata: sample_rate=%s, num_channels=%s, num_frames=%s, bits_per_sample=%s, encoding=%s",
            metadata.sample_rate,
            metadata.num_channels,
            metadata.num_frames,
            metadata.bits_per_sample,
            metadata.encoding,
        )
    except Exception as error:
        logger.info("Torchaudio could not read file metadata: %s", error)


def format_transcription_segments(segments: list[object]) -> list[dict[str, object]]:
    output_segments = []
    for segment in segments:
        output_segment = {
            "avg_logprob": segment.avg_logprob,
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text,
            "words": [],
        }
        if segment.words is not None:
            output_segment["words"] = [
                {
                    "start": float(word.start),
                    "end": float(word.end),
                    "word": word.word,
                    "probability": word.probability,
                }
                for word in segment.words
            ]
        output_segments.append(output_segment)
    return output_segments


def post_process_segments(
    segments: list[dict[str, object]], diarization_list: object
) -> list[dict[str, object]]:
    diarize_segments = []
    for turn, speaker in diarization_list:
        diarize_segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    diarize_df = pd.DataFrame(diarize_segments)
    final_segments = []
    for segment in segments:
        diarize_df["intersection"] = np.minimum(
            diarize_df["end"], segment["end"]
        ) - np.maximum(diarize_df["start"], segment["start"])
        dia_tmp = diarize_df[diarize_df["intersection"] > 0]
        speaker = "UNKNOWN"
        if len(dia_tmp) > 0:
            speaker = (
                dia_tmp.groupby("speaker")["intersection"]
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )

        words_with_speakers = []
        for word in segment["words"]:
            diarize_df["intersection"] = np.minimum(
                diarize_df["end"], word["end"]
            ) - np.maximum(diarize_df["start"], word["start"])
            dia_tmp = diarize_df[diarize_df["intersection"] > 0]
            word_speaker = speaker
            if len(dia_tmp) > 0:
                word_speaker = (
                    dia_tmp.groupby("speaker")["intersection"]
                    .sum()
                    .sort_values(ascending=False)
                    .index[0]
                )
            word["speaker"] = word_speaker
            words_with_speakers.append(word)

        final_segments.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": speaker,
                "avg_logprob": segment["avg_logprob"],
                "words": words_with_speakers,
            }
        )

    if len(final_segments) == 0:
        return final_segments

    grouped_segments = []
    current_group = final_segments[0].copy()
    sentence_end_pattern = r"[.!?]+"

    for segment in final_segments[1:]:
        time_gap = segment["start"] - current_group["end"]
        current_duration = current_group["end"] - current_group["start"]
        can_combine = (
            segment["speaker"] == current_group["speaker"]
            and time_gap <= 1.0
            and current_duration < 30.0
            and not re.search(sentence_end_pattern, current_group["text"][-1:])
        )
        if can_combine:
            current_group["end"] = segment["end"]
            current_group["text"] += " " + segment["text"]
            current_group["words"].extend(segment["words"])
            continue

        grouped_segments.append(current_group)
        current_group = segment.copy()

    grouped_segments.append(current_group)

    for segment in grouped_segments:
        segment["text"] = re.sub(r"\s+", " ", segment["text"]).strip()
        segment["text"] = re.sub(r"\s+([.,!?])", r"\1", segment["text"])
        segment["duration"] = segment["end"] - segment["start"]

    return grouped_segments


def get_gpu_type() -> str:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"]
        )
        return output.decode().strip()
    except Exception:
        return "Unknown"
