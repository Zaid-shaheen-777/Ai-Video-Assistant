import unittest
from pathlib import Path

from core.transcriber import prepare_upload_path


class PrepareUploadPathTests(unittest.TestCase):
    def test_sanitizes_non_ascii_filename(self):
        temp_dir = Path(__file__).resolve().parent
        source_path = temp_dir / "video｜title.wav"
        source_path.write_bytes(b"audio")

        try:
            safe_path = prepare_upload_path(str(source_path))
            self.assertTrue(Path(safe_path).exists())
            self.assertEqual(Path(safe_path).name, "video_title.wav")
        finally:
            if source_path.exists():
                source_path.unlink()
            safe_path = temp_dir / "video_title.wav"
            if safe_path.exists():
                safe_path.unlink()


if __name__ == "__main__":
    unittest.main()
