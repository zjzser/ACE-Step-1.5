"""Tests for TensorBoard sample-preview helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torchaudio

from acestep.training_v2.tensorboard_preview import (
    build_sample_preview,
    build_spectrogram_image,
    extract_first_audio_path,
    load_audio_preview,
    should_log_sample_preview,
)


class TestShouldLogSamplePreview(unittest.TestCase):
    """Validate preview cadence decisions."""

    def test_returns_false_when_disabled(self) -> None:
        self.assertFalse(should_log_sample_preview(0, 0))

    def test_returns_true_on_matching_epoch(self) -> None:
        self.assertTrue(should_log_sample_preview(2, 1))

    def test_returns_false_on_non_matching_epoch(self) -> None:
        self.assertFalse(should_log_sample_preview(3, 1))


class TestBuildSamplePreview(unittest.TestCase):
    """Validate TensorBoard preview formatting."""

    def test_formats_metadata_and_shapes(self) -> None:
        batch = {
            "target_latents": torch.zeros(2, 8, 64),
            "encoder_hidden_states": torch.zeros(2, 12, 1024),
            "metadata": [
                {
                    "filename": "demo.wav",
                    "caption": "lofi piano",
                    "genre": "jazz",
                    "lyrics": "line one\nline two",
                    "audio_path": "/tmp/demo.wav",
                },
            ],
        }

        preview = build_sample_preview(batch, epoch=0)

        self.assertIn("epoch: 1", preview)
        self.assertIn("target_latents_shape: (2, 8, 64)", preview)
        self.assertIn("encoder_hidden_states_shape: (2, 12, 1024)", preview)
        self.assertIn("filename: demo.wav", preview)
        self.assertIn("caption: lofi piano", preview)
        self.assertIn("lyrics: line one line two", preview)

    def test_handles_missing_metadata(self) -> None:
        preview = build_sample_preview({"metadata": None}, epoch=1)

        self.assertIn("epoch: 2", preview)
        self.assertIn("samples: metadata unavailable", preview)


class TestAudioPreviewHelpers(unittest.TestCase):
    """Validate audio preview extraction and spectrogram generation."""

    def test_extract_first_audio_path(self) -> None:
        batch = {"metadata": [{"audio_path": "/tmp/demo.wav"}]}
        self.assertEqual("/tmp/demo.wav", extract_first_audio_path(batch))

    def test_load_audio_preview_resamples_and_mixes_down(self) -> None:
        with TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "demo.wav"
            waveform = torch.stack([
                torch.linspace(-0.5, 0.5, 8000),
                torch.linspace(0.5, -0.5, 8000),
            ])
            torchaudio.save(str(audio_path), waveform, 8000)

            preview, sample_rate = load_audio_preview(str(audio_path), target_sample_rate=16000)

            self.assertEqual(16000, sample_rate)
            self.assertEqual(1, preview.shape[0])
            self.assertGreater(preview.shape[1], 8000)

    def test_build_spectrogram_image_returns_single_channel_image(self) -> None:
        waveform = torch.sin(torch.linspace(0, 100, 16000)).unsqueeze(0)

        image = build_spectrogram_image(waveform)

        self.assertEqual(3, image.ndim)
        self.assertEqual(1, image.shape[0])
        self.assertGreater(image.shape[1], 0)
        self.assertGreater(image.shape[2], 0)
