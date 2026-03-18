import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import gc
import wave
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import noisereduce as nr
import numpy as np
import torch
import whisperx
from pydub import AudioSegment
from pydub.effects import normalize
from tqdm import tqdm


@dataclass(frozen=True)
class DeviceConfig:
    device: str
    compute_type: str
    backend: str


class HardwareSelector:
    """Selects execution backend, preferring AMD ROCm for RX5700 when available."""

    def __init__(self, preferred_gpu_name: str = "RX5700"):
        self.preferred_gpu_name = preferred_gpu_name.lower()

    def select(self) -> DeviceConfig:
        if not torch.cuda.is_available():
            return DeviceConfig(device="cpu", compute_type="int8", backend="cpu")

        device_name = torch.cuda.get_device_name(0).lower()
        is_rocm = bool(getattr(torch.version, "hip", None))

        if is_rocm:
            compute_type = "float16"
            backend = "rocm"
        else:
            compute_type = "float16"
            backend = "cuda"

        if self.preferred_gpu_name not in device_name:
            print(
                f"Warning: preferred GPU '{self.preferred_gpu_name}' not detected. "
                f"Using '{device_name}' via {backend}."
            )
        else:
            print(f"Using AMD GPU '{device_name}' via {backend}.")

        # whisperx expects device='cuda' for both CUDA and ROCm torch backends.
        return DeviceConfig(device="cuda", compute_type=compute_type, backend=backend)


class AudioProcessor:
    def __init__(self, target_sample_rate: int = 16000, noise_reduction_amount: float = 0.75):
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


def load_wav_mono_16k(audio_path: str, target_sample_rate: int = 16000) -> np.ndarray:
    """
    Load mono 16-bit PCM WAV without invoking ffmpeg.
    This avoids whisperx.load_audio subprocess dependency on ffmpeg executables.
    """
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


class WhisperXPipeline:
    def __init__(self, hf_token: str, device_config: DeviceConfig):
        self.hf_token = hf_token
        self.device_config = device_config

    @staticmethod
    def _free_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transcribe_and_align(self, audio_path: str, batch_size: int = 8) -> Dict[str, Any]:
        model = whisperx.load_model(
            "large-v3",
            self.device_config.device,
            compute_type=self.device_config.compute_type,
        )
        audio = load_wav_mono_16k(audio_path)

        result: Dict[str, Any] = model.transcribe(audio, batch_size=batch_size)
        self._free_memory()

        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=self.device_config.device,
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            self.device_config.device,
            return_char_alignments=False,
        )
        self._free_memory()
        return result


class InterviewTranscriptionService:
    """Independent transcription pipeline (audio prep -> ASR -> align)."""

    def __init__(self, hf_token: str, preferred_gpu_name: str = "RX5700"):
        self.audio_processor = AudioProcessor()
        self.device_config = HardwareSelector(preferred_gpu_name).select()
        self.whisper_pipeline = WhisperXPipeline(hf_token, self.device_config)

    def transcribe_interview(self, audio_path: str, test_duration_sec: Optional[int] = None) -> List[Dict[str, Any]]:
        cleaned_path = self.audio_processor.process_file(audio_path, test_duration_sec=test_duration_sec)

        with tqdm(total=2, desc="Transcription Pipeline", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            pbar.set_description("Transcribing & Aligning Audio")
            transcript_result = self.whisper_pipeline.transcribe_and_align(cleaned_path)
            pbar.update(1)

            pbar.set_description("Finalizing")
            pbar.update(1)

        return transcript_result["segments"]


def main() -> None:
    access_token = os.getenv("HF_ACCESS_TOKEN")
    if not access_token:
        access_token = "e6itqaPMdDfti6Gn3F75BSGQ6vegXBb7ADBpQkpq32A="
        
    interviews_at = r"D:\DIP24\DIP-Statistical-Analysis\transcription_model\interviews"
    
    print("Loading AI models into RAM... This may take a moment.")
    service = InterviewTranscriptionService(hf_token=access_token, preferred_gpu_name="RX5700")
    
    for root, dirs, files in os.walk(interviews_at):
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

            if file is None:
                raise ValueError("Audio path is None — check earlier step")

            try:
                segments = service.transcribe_interview(audio_file)
                with open(output_path, "w", encoding="utf-8") as text_file:
                    for segment in segments:
                        start = round(segment.get("start", 0), 2)
                        end = round(segment.get("end", 0), 2)
                        text = segment.get("text", "").strip()
                        text_file.write(f"[{start}s - {end}]s {text}\n")
                print(f"Transcription saved to: {output_path}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    main()