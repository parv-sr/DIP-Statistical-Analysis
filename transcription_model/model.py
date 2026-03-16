from transformers import pipeline
import torch
import gc
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import whisperx
import os
import numpy as np
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize

class AudioProcessor:
    def __init__(self, target_sample_rate: int = 16000, noise_reduction_amount: float = 0.75):
        self.target_sr = target_sample_rate
        self.nr_amount = noise_reduction_amount

    def process_file(self, input_path: str, output_path: Optional[str] = None,
                     test_duration_sec: Optional[int] = None) -> str:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Could not find the file: {input_path}")

        if output_path is None:
            base_name, _ = os.path.splitext(input_path)
            output_path = f"{base_name}_cleaned.wav"

        print(f"Loading and formatting {input_path}...")
        audio: AudioSegment = AudioSegment.from_file(input_path)


        if test_duration_sec is not None:
            print(f"Slicing audio to the first {test_duration_sec} seconds for rapid testing...")
            # pydub works in milliseconds, so multiply by 1000
            audio = audio[:test_duration_sec * 1000]

        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(self.target_sr)

        print("Applying spectral noise reduction...")
        samples: np.ndarray = np.array(audio.get_array_of_samples())
        samples_float: np.ndarray = samples.astype(np.float32) / 32768.0

        cleaned_float: np.ndarray = nr.reduce_noise(
            y=samples_float,
            sr=self.target_sr,
            prop_decrease=self.nr_amount
        )

        cleaned_int16: np.ndarray = (cleaned_float * 32768.0).astype(np.int16)
        audio = audio._spawn(cleaned_int16.tobytes())

        print("Normalizing audio levels...")
        audio = normalize(audio)

        print(f"Exporting final master to {output_path}...")
        audio.export(output_path, format="wav")

        return output_path
class Transcriber:
    """
    An OOP based pipeline for transcribing DIP Group 24's interviews.
    Authored by: Parv (Group 24)
    """

    def __init__(self, hf_token: str):
        self.hf_token: str = hf_token

        if torch.cuda.is_available():
            self.device: str = "cuda"
            self.compute_type: str = "float16"
            print("NVIDIA GPU detected. Running in high-speed CUDA mode.")
        else:
            self.device: str = "cpu"
            self.compute_type: str = "int8"
            print("No NVIDIA GPU detected. Falling back to CPU mode (this will be slow).")

    def free_vram(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transcribe_and_align(self, audio_path: str, batch_size: int = 8) -> Dict[str, Any]:
        model: Any = whisperx.load_model("large-v3", self.device, compute_type=self.compute_type)
        audio: Any = whisperx.load_audio(audio_path)

        result: Dict[str, Any] = model.transcribe(audio, batch_size=batch_size)
        self.free_vram()

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
        self.free_vram()

        return result

    def diarize(self, audio_path: str) -> Any:
        """Handles speaker clustering using Pyannote."""
        audio = whisperx.load_audio(audio_path)
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)
        diarize_segments = diarize_model(audio)
        self.free_vram()
        
        return diarize_segments

    def process_interview(self, audio_path: str) -> List[Dict[str, Any]]:
        with tqdm(total=4, desc="Processing Pipeline", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            pbar.set_description("Transcribing Audio")
            transcript_result: Dict[str, Any] = self.transcribe_and_align(audio_path)
            pbar.update(1)  # Step 1

            pbar.set_description("Clustering Speakers")
            diarization_result = self.diarize(audio_path)
            pbar.update(1)  # Step 2

            pbar.set_description("Merging Timestamps")
            final_result: Dict[str, Any] = whisperx.assign_word_speakers(diarization_result, transcript_result)
            pbar.update(1)  # Step 3

            pbar.set_description("Finalizing")
            pbar.update(1)  # Step 4

        return final_result["segments"]


if __name__ == "__main__":
    ACCESS_TOKEN = "e6itqaPMdDfti6Gn3F75BSGQ6vegXBb7ADBpQkpq32A="
    AUDIO_FILE = r"C:\F DRIVE\Python\DIP Statistical analysis\transcription_model\Interviews\navdeep_interview-01.wav"

    preprocessor = AudioProcessor()
    transcriber = Transcriber(ACCESS_TOKEN)
    AUDIO_FILE = preprocessor.process_file(AUDIO_FILE, test_duration_sec=60)
    segments = transcriber.process_interview(AUDIO_FILE)

    with open(f"{AUDIO_FILE}_transcription.txt", 'x') as f:
        print(f"Created transcript file: {AUDIO_FILE}.\nTranscribing...")

        for segment in segments:
            speaker: str = segment.get("speaker", "UNKNOWN")
            text: str = segment.get("text", "").strip()

            print(f"[{speaker}]: {text}")
            f.write(f"[{speaker}]: {text}")