"""Tests for generated TensorBoard preview helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from acestep.training_v2.tensorboard_generated_preview import (
    build_generated_preview_text,
    build_generation_preview_request,
)


class TestBuildGenerationPreviewRequest(unittest.TestCase):
    """Validate metadata-to-generation-request mapping."""

    def test_builds_request_from_metadata(self) -> None:
        cfg = SimpleNamespace(num_inference_steps=12, shift=3.0, seed=123)
        metadata = {
            'caption': 'warm lofi beat',
            'lyrics': 'hello world',
            'bpm': '120',
            'keyscale': 'C major',
            'timesignature': '4/4',
            'duration': 42,
            'language': 'en',
        }

        request = build_generation_preview_request(metadata, cfg)

        self.assertEqual('warm lofi beat', request['caption'])
        self.assertEqual('hello world', request['lyrics'])
        self.assertEqual(120, request['bpm'])
        self.assertEqual('C major', request['key_scale'])
        self.assertEqual('4/4', request['time_signature'])
        self.assertEqual(30.0, request['audio_duration'])
        self.assertEqual(12, request['inference_steps'])
        self.assertEqual(3.0, request['shift'])
        self.assertEqual(123, request['seed'])

    def test_uses_safe_defaults(self) -> None:
        cfg = SimpleNamespace(num_inference_steps=8, shift=1.0, seed=42)
        request = build_generation_preview_request({'filename': 'demo.wav'}, cfg)

        self.assertEqual('demo.wav', request['caption'])
        self.assertEqual('[Instrumental]', request['lyrics'])
        self.assertEqual(30.0, request['audio_duration'])
        self.assertEqual('en', request['vocal_language'])


class TestBuildGeneratedPreviewText(unittest.TestCase):
    """Validate TensorBoard text formatting for generated previews."""

    def test_includes_key_fields(self) -> None:
        text = build_generated_preview_text({
            'caption': 'test caption',
            'lyrics': '[Instrumental]',
            'audio_duration': 15.0,
            'inference_steps': 8,
            'shift': 3.0,
            'seed': 7,
            'bpm': None,
            'key_scale': '',
            'time_signature': '',
            'vocal_language': 'en',
        })

        self.assertIn('## Generated Preview', text)
        self.assertIn('- generated_from_current_model: `true`', text)
        self.assertIn('- caption: `test caption`', text)
        self.assertIn('- duration: `15.0`', text)
        self.assertIn('- seed: `7`', text)
