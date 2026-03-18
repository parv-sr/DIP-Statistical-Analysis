import gc
import os
import re
import wave
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import noisereduce as nr
import numpy as np
import torch
import whisperx
from pydub import AudioSegment
from pydub.effects import normalize
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

TARGET_SAMPLE_RATE = 16000
ENGLISH_MODEL_NAME = "large-v3"
INDIC_MODEL_NAME = "parthiv11/indic_whisper_hi_multi_gpu"
LANGUAGE_MODE_AUTO = "auto"
LANGUAGE_MODE_ENGLISH = "english"
LANGUAGE_MODE_INDIC = "indic"

INDIC_LANGUAGE_CODES = {
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml",
    "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
}

REPLACEMENTS = {
    "mera naam": "मेरा नाम",
    "hai na": "है ना",
    "kya": "क्या",
    "haan": "हाँ",
    "nahi": "नहीं",
    "kyunki": "क्योंकि",
}


@dataclass(frozen=True)
class DeviceConfig:
    device: str
    compute_type: str
    backend: str
    device_index: int


@dataclass
class RoutedChunk:
    start: float
    end: float
    route: str
    detector_language: str
    text_hint: str


class HardwareSelector:
    """Select a CUDA execution backend and fail fast when no NVIDIA GPU is available."""

    def select(self) -> DeviceConfig:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was not detected. This transcription pipeline is configured for NVIDIA GPU inference."
            )

        device_name = torch.cuda.get_device_name(0)
        print(f"Using NVIDIA GPU '{device_name}' via CUDA.")
        return DeviceConfig(device="cuda", compute_type="float16", backend="cuda", device_index=0)


class AudioProcessor:
    def __init__(self, target_sample_rate: int = TARGET_SAMPLE_RATE, noise_reduction_amount: float = 0.35):
        self.target_sr = target_sample_rate
        self.nr_amount = noise_reduction_amount

    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        test_duration_sec: Optional[int] = None,
    ) -> str:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Could not find the file: {input_path}")

        if output_path is None:
            base_name, _ = os.path.splitext(input_path)
            output_path = f"{base_name}_cleaned.wav"

        audio = AudioSegment.from_file(input_path)

        if test_duration_sec is not None:
            audio = audio[: test_duration_sec * 1000]

        audio = audio.set_channels(1).set_frame_rate(self.target_sr)
        samples = np.array(audio.get_array_of_samples())
        samples_float = samples.astype(np.float32) / 32768.0

        cleaned_float = nr.reduce_noise(
            y=samples_float,
            sr=self.target_sr,
            prop_decrease=self.nr_amount,
        )

        cleaned_int16 = np.clip(cleaned_float * 32768.0, -32768, 32767).astype(np.int16)
        cleaned_audio = audio._spawn(cleaned_int16.tobytes())
        cleaned_audio = normalize(cleaned_audio)
        cleaned_audio.export(output_path, format="wav")
        return output_path


def load_wav_mono_16k(audio_path: str, target_sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load mono 16-bit PCM WAV without invoking ffmpeg."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with wave.open(audio_path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(
            f"Expected 16-bit PCM WAV after preprocessing, got sample width={sample_width} bytes."
        )
    if sample_rate != target_sample_rate:
        raise ValueError(
            f"Expected sample rate {target_sample_rate}Hz after preprocessing, got {sample_rate}Hz."
        )

    samples = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)

    audio = samples.astype(np.float32) / 32768.0
    return np.clip(audio, -1.0, 1.0)


def extract_audio_window(audio: np.ndarray, start_sec: float, end_sec: float) -> np.ndarray:
    start_idx = max(0, int(start_sec * TARGET_SAMPLE_RATE))
    end_idx = min(audio.shape[0], int(end_sec * TARGET_SAMPLE_RATE))
    return np.copy(audio[start_idx:end_idx])


class HindiPostProcessor:
    def __init__(self, replacements: Dict[str, str]):
        self.replacements = {
            key.lower(): value for key, value in replacements.items()
        }

    def normalize(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        for source, target in self.replacements.items():
            normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\s+([,.!?।])", r"\1", normalized)
        return normalized.strip()


class LanguageRouter:
    def __init__(self, language_mode: str = LANGUAGE_MODE_AUTO):
        language_mode = (language_mode or LANGUAGE_MODE_AUTO).strip().lower()
        if language_mode not in {LANGUAGE_MODE_AUTO, LANGUAGE_MODE_ENGLISH, LANGUAGE_MODE_INDIC}:
            raise ValueError(
                f"Unsupported language mode '{language_mode}'. Use auto, english, or indic."
            )
        self.language_mode = language_mode

    @staticmethod
    def _contains_devanagari(text: str) -> bool:
        return any("\u0900" <= char <= "\u097F" for char in text)

    def choose_route(self, detector_language: str, text_hint: str, global_language: str) -> str:
        if self.language_mode == LANGUAGE_MODE_ENGLISH:
            return LANGUAGE_MODE_ENGLISH
        if self.language_mode == LANGUAGE_MODE_INDIC:
            return LANGUAGE_MODE_INDIC

        detector_language = (detector_language or "").lower()
        global_language = (global_language or "").lower()
        text_hint_lower = (text_hint or "").lower()

        if detector_language in INDIC_LANGUAGE_CODES:
            return LANGUAGE_MODE_INDIC
        if detector_language == "en":
            if self._contains_devanagari(text_hint) or any(key in text_hint_lower for key in REPLACEMENTS):
                return LANGUAGE_MODE_INDIC
            return LANGUAGE_MODE_ENGLISH
        if self._contains_devanagari(text_hint) or any(key in text_hint_lower for key in REPLACEMENTS):
            return LANGUAGE_MODE_INDIC
        if global_language in INDIC_LANGUAGE_CODES:
            return LANGUAGE_MODE_INDIC
        return LANGUAGE_MODE_ENGLISH


class WhisperXPipeline:
    def __init__(self, hf_token: str, device_config: DeviceConfig):
        self.hf_token = hf_token
        self.device_config = device_config
        self._detector = None
        self._english = None
        self._english_align_model = None
        self._english_align_metadata = None

    @staticmethod
    def _free_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_detector(self):
        if self._detector is None:
            self._detector = whisperx.load_model(
                ENGLISH_MODEL_NAME,
                self.device_config.device,
                compute_type=self.device_config.compute_type,
                vad_method="silero",
                language=None,
                use_auth_token=self.hf_token,
                asr_options={
                    "beam_size": 8,
                    "best_of": 8,
                    "condition_on_previous_text": True,
                },
            )
        return self._detector

    def _load_english(self):
        if self._english is None:
            self._english = whisperx.load_model(
                ENGLISH_MODEL_NAME,
                self.device_config.device,
                compute_type=self.device_config.compute_type,
                vad_method="silero",
                language="en",
                use_auth_token=self.hf_token,
                asr_options={
                    "beam_size": 8,
                    "best_of": 8,
                    "condition_on_previous_text": True,
                },
            )
        return self._english

    def _load_english_aligner(self):
        if self._english_align_model is None or self._english_align_metadata is None:
            self._english_align_model, self._english_align_metadata = whisperx.load_align_model(
                language_code="en",
                device=self.device_config.device,
            )
        return self._english_align_model, self._english_align_metadata

    def detect_language(self, audio: np.ndarray) -> str:
        return self._load_detector().detect_language(audio)

    def detect_segments(self, audio: np.ndarray, batch_size: int = 4) -> Dict[str, Any]:
        return self._load_detector().transcribe(audio, batch_size=batch_size, language=None)

    def transcribe_english_chunk(self, audio: np.ndarray, start_offset: float) -> List[Dict[str, Any]]:
        result = self._load_english().transcribe(audio, batch_size=1, language="en")
        segments: List[Dict[str, Any]] = []
        for segment in result["segments"]:
            segments.append(
                {
                    "start": round(start_offset + float(segment.get("start", 0.0)), 3),
                    "end": round(start_offset + float(segment.get("end", 0.0)), 3),
                    "text": segment.get("text", "").strip(),
                    "language": "en",
                    "route": LANGUAGE_MODE_ENGLISH,
                }
            )
        return segments

    def align_english_segments(self, audio: np.ndarray, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        segments = [segment for segment in segments if segment.get("text", "").strip()]
        if not segments:
            return []

        model_a, metadata = self._load_english_aligner()
        aligned = whisperx.align(
            segments,
            model_a,
            metadata,
            audio,
            self.device_config.device,
            return_char_alignments=False,
        )
        return aligned["segments"]

    def unload_detector(self) -> None:
        self._detector = None
        self._free_memory()


class IndicWhisperPipeline:
    def __init__(self, device_config: DeviceConfig):
        self.device_config = device_config
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            INDIC_MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.to(self.device_config.device)
        processor = AutoProcessor.from_pretrained(INDIC_MODEL_NAME)
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
            device=self.device_config.device_index,
        )
        return self._pipeline

    def transcribe_chunk(self, audio: np.ndarray, start_offset: float, detected_language: str) -> List[Dict[str, Any]]:
        recognizer = self._load_pipeline()
        result = recognizer(
            {"array": audio, "sampling_rate": TARGET_SAMPLE_RATE},
            return_timestamps=True,
            generate_kwargs={"task": "transcribe"},
        )

        chunks = result.get("chunks") or []
        if not chunks:
            return [
                {
                    "start": round(start_offset, 3),
                    "end": round(start_offset + (len(audio) / TARGET_SAMPLE_RATE), 3),
                    "text": result.get("text", "").strip(),
                    "language": detected_language or "indic",
                    "route": LANGUAGE_MODE_INDIC,
                }
            ]

        segments: List[Dict[str, Any]] = []
        for chunk in chunks:
            timestamp = chunk.get("timestamp") or (None, None)
            chunk_start = start_offset if timestamp[0] is None else start_offset + float(timestamp[0])
            chunk_end = start_offset + (len(audio) / TARGET_SAMPLE_RATE) if timestamp[1] is None else start_offset + float(timestamp[1])
            segments.append(
                {
                    "start": round(chunk_start, 3),
                    "end": round(chunk_end, 3),
                    "text": chunk.get("text", "").strip(),
                    "language": detected_language or "indic",
                    "route": LANGUAGE_MODE_INDIC,
                }
            )
        return segments


class InterviewTranscriptionService:
    """Independent transcription pipeline with segment-level language routing."""

    def __init__(self, hf_token: str):
        self.audio_processor = AudioProcessor()
        self.device_config = HardwareSelector().select()
        self.router = LanguageRouter(os.getenv("TRANSCRIPTION_LANGUAGE_MODE", LANGUAGE_MODE_AUTO))
        self.post_processor = HindiPostProcessor(REPLACEMENTS)
        self.whisper_pipeline = WhisperXPipeline(hf_token, self.device_config)
        self.indic_pipeline = IndicWhisperPipeline(self.device_config)

    @staticmethod
    def _merge_chunks(chunks: Sequence[RoutedChunk], max_gap_sec: float = 0.35) -> List[RoutedChunk]:
        if not chunks:
            return []

        merged: List[RoutedChunk] = [RoutedChunk(**vars(chunks[0]))]
        for chunk in chunks[1:]:
            previous = merged[-1]
            should_merge = (
                previous.route == chunk.route
                and previous.detector_language == chunk.detector_language
                and chunk.start - previous.end <= max_gap_sec
            )
            if should_merge:
                previous.end = chunk.end
                previous.text_hint = f"{previous.text_hint} {chunk.text_hint}".strip()
            else:
                merged.append(RoutedChunk(**vars(chunk)))
        return merged

    def _build_routed_chunks(self, audio: np.ndarray, coarse_segments: Sequence[Dict[str, Any]], global_language: str) -> List[RoutedChunk]:
        routed_chunks: List[RoutedChunk] = []
        for segment in coarse_segments:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", 0.0))
            text_hint = segment.get("text", "").strip()
            segment_audio = extract_audio_window(audio, start, end)

            if segment_audio.size == 0:
                continue

            detector_language = global_language
            if end - start >= 1.0:
                detector_language = self.whisper_pipeline.detect_language(segment_audio)

            route = self.router.choose_route(detector_language, text_hint, global_language)
            routed_chunks.append(
                RoutedChunk(
                    start=start,
                    end=end,
                    route=route,
                    detector_language=detector_language,
                    text_hint=text_hint,
                )
            )
        return self._merge_chunks(routed_chunks)

    def _post_process(self, segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        for segment in sorted(segments, key=lambda item: (item.get("start", 0.0), item.get("end", 0.0))):
            updated = dict(segment)
            if updated.get("route") == LANGUAGE_MODE_INDIC:
                updated["text"] = self.post_processor.normalize(updated.get("text", ""))
            else:
                updated["text"] = re.sub(r"\s+", " ", updated.get("text", "")).strip()
            processed.append(updated)
        return processed

    def transcribe_interview(
        self,
        audio_path: str,
        test_duration_sec: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        cleaned_path = self.audio_processor.process_file(audio_path, test_duration_sec=test_duration_sec)
        audio = load_wav_mono_16k(cleaned_path)

        with tqdm(total=4, desc="Transcription Pipeline", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            pbar.set_description("Detecting speech and language")
            coarse_result = self.whisper_pipeline.detect_segments(audio)
            pbar.update(1)

            routed_chunks = self._build_routed_chunks(audio, coarse_result["segments"], coarse_result["language"])
            self.whisper_pipeline.unload_detector()
            pbar.set_description("Routing segments")
            pbar.update(1)

            english_segments: List[Dict[str, Any]] = []
            indic_segments: List[Dict[str, Any]] = []
            for chunk in routed_chunks:
                chunk_audio = extract_audio_window(audio, chunk.start, chunk.end)
                if chunk.route == LANGUAGE_MODE_ENGLISH:
                    english_segments.extend(self.whisper_pipeline.transcribe_english_chunk(chunk_audio, chunk.start))
                else:
                    indic_segments.extend(
                        self.indic_pipeline.transcribe_chunk(chunk_audio, chunk.start, chunk.detector_language)
                    )
            pbar.set_description("Transcribing routed chunks")
            pbar.update(1)

            aligned_english = self.whisper_pipeline.align_english_segments(audio, english_segments)
            pbar.set_description("Finalizing transcript")
            pbar.update(1)

        return self._post_process([*aligned_english, *indic_segments])


def _resolve_access_token() -> str:
    access_token = os.getenv("HF_ACCESS_TOKEN")
    if access_token:
        return access_token
    return "e6itqaPMdDfti6Gn3F75BSGQ6vegXBb7ADBpQkpq32A="


def main() -> None:
    access_token = _resolve_access_token()
    interviews_at = os.getenv(
        "INTERVIEW_AUDIO_DIR",
        r"D:\DIP24\DIP-Statistical-Analysis\transcription_model\interviews",
    )

    print("Loading AI models into GPU memory... This may take a moment.")
    service = InterviewTranscriptionService(hf_token=access_token)

    for root, _, files in os.walk(interviews_at):
        for file in files:
            if not file.lower().endswith(".wav"):
                continue
            if file.lower().endswith("_cleaned.wav"):
                continue

            audio_file = os.path.join(root, file)
            output_path = f"{audio_file}_transcription.txt"

            if os.path.exists(output_path):
                print(f"Transcription already exists for {file}, skipping.")
                continue

            print(f"Processing interview audio: {file}")
            try:
                segments = service.transcribe_interview(audio_file)
                with open(output_path, "w", encoding="utf-8") as text_file:
                    for segment in segments:
                        start = round(float(segment.get("start", 0.0)), 2)
                        end = round(float(segment.get("end", 0.0)), 2)
                        language = segment.get("language", "unknown")
                        route = segment.get("route", "unknown")
                        text = segment.get("text", "").strip()
                        text_file.write(f"[{start}s - {end}s] ({route}/{language}) {text}\n")
                print(f"Transcription saved to: {output_path}")
            except Exception as exc:
                print(f"Error processing {file}: {exc}")


def process_one() -> None:
    access_token = _resolve_access_token()
    audio_file = os.getenv("INTERVIEW_AUDIO_FILE")

    if not audio_file:
        raise RuntimeError("Set INTERVIEW_AUDIO_FILE before running process_one().")

    service = InterviewTranscriptionService(hf_token=access_token)
    segments = service.transcribe_interview(audio_file, test_duration_sec=60)

    output_path = f"{audio_file}_transcription.txt"
    with open(output_path, "w", encoding="utf-8") as file:
        for segment in segments:
            start = round(float(segment.get("start", 0.0)), 2)
            end = round(float(segment.get("end", 0.0)), 2)
            language = segment.get("language", "unknown")
            route = segment.get("route", "unknown")
            text = segment.get("text", "").strip()
            file.write(f"[{start}s - {end}s] ({route}/{language}) {text}\n")

    print(f"Created transcript file: {output_path}")


if __name__ == "__main__":
    main()