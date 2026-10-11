from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import main


class StackMathTests(unittest.TestCase):
    def test_mean_and_median_preserve_uint16(self) -> None:
        with tempfile.TemporaryDirectory(prefix="moon-stack-test-") as temp:
            root = Path(temp)
            values = np.array(
                [
                    np.full((12, 10), 100, dtype=np.uint16),
                    np.full((12, 10), 200, dtype=np.uint16),
                    np.full((12, 10), 900, dtype=np.uint16),
                ]
            )
            np.save(root / "frames.npy", values)

            main.stack_frames(root / "frames.npy", root / "mean.tif", "mean", 100, 0, lambda _: None)
            main.stack_frames(
                root / "frames.npy", root / "median.tif", "median", 100, 0, lambda _: None
            )

            mean = main.read_image(root / "mean.tif")
            median = main.read_image(root / "median.tif")
            self.assertEqual(mean.dtype, np.uint16)
            self.assertEqual(median.dtype, np.uint16)
            np.testing.assert_array_equal(mean, np.full((12, 10), 400, dtype=np.uint16))
            np.testing.assert_array_equal(median, np.full((12, 10), 200, dtype=np.uint16))

    def test_local_stack_reduces_smooth_spatial_distortion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="moon-local-test-") as temp:
            root = Path(temp)
            height = width = 176
            yy, xx = np.mgrid[:height, :width].astype(np.float32)
            radius = 70
            disc = (xx - width / 2) ** 2 + (yy - height / 2) ** 2 <= radius**2
            rng = np.random.default_rng(20261011)
            random_texture = cv2.GaussianBlur(
                rng.normal(0, 1, (height, width)).astype(np.float32), (0, 0), 2.0
            )
            random_texture = (random_texture - random_texture.min()) / (
                random_texture.max() - random_texture.min()
            )
            texture = (
                16000
                + 28000 * random_texture
                + 2500 * np.sin(xx / 13.0)
                + 1800 * np.cos(yy / 17.0)
            )
            base = np.zeros((height, width), dtype=np.uint16)
            base[disc] = np.clip(texture[disc], 1000, 62000).astype(np.uint16)
            cv2.circle(base, (70, 72), 10, 15500, -1)
            cv2.circle(base, (111, 105), 14, 39000, -1)
            cv2.circle(base, (100, 60), 6, 18000, -1)

            frame_count = 10
            frames = np.lib.format.open_memmap(
                root / "frames.npy",
                mode="w+",
                dtype=np.uint16,
                shape=(frame_count, height, width),
            )
            for index in range(frame_count):
                phase = 2.0 * np.pi * index / frame_count
                dx = 3.0 * np.sin(2.0 * np.pi * yy / height * 2.0 + phase)
                dx += 0.7 * np.sin(2.0 * np.pi * xx / width * 3.0 - phase)
                dy = 2.7 * np.sin(2.0 * np.pi * xx / width * 2.0 - phase)
                dy += 0.6 * np.sin(2.0 * np.pi * yy / height * 3.0 + phase)
                warped = cv2.remap(
                    base,
                    xx + dx.astype(np.float32),
                    yy + dy.astype(np.float32),
                    cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT,
                )
                noisy = warped.astype(np.float32) + rng.normal(0, 45, warped.shape)
                frames[index] = np.clip(noisy, 0, 65535).astype(np.uint16)
            frames.flush()
            del frames

            main.stack_frames(root / "frames.npy", root / "mean.tif", "mean", 60, 48, lambda _: None)
            main.stack_frames(
                root / "frames.npy", root / "local.tif", "local", 60, 48, lambda _: None
            )
            mean = main.read_image(root / "mean.tif").astype(np.float32)
            local = main.read_image(root / "local.tif").astype(np.float32)
            inner = (xx - width / 2) ** 2 + (yy - height / 2) ** 2 <= (radius * 0.78) ** 2
            mean_error = float(np.mean((mean[inner] - base.astype(np.float32)[inner]) ** 2))
            local_error = float(np.mean((local[inner] - base.astype(np.float32)[inner]) ** 2))

            self.assertEqual(local.shape, base.shape)
            self.assertEqual(local.dtype, np.float32)
            self.assertLess(local_error, mean_error * 0.90)
            self.assertTrue((root / "local_alignment_points.png").is_file())

    def test_processing_report_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="moon-report-test-") as temp:
            root = Path(temp)
            scan = main.ScanResult(root / "reference.tif", 640, 480, (320.0, 240.0), 100.0, 1.25, 1.0)
            main.save_processing_report(
                root,
                root / "input",
                scan,
                12,
                9,
                (20, 30, 420, 430),
                "local",
                50,
                64,
                False,
            )
            report = json.loads((root / "processing_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["version"], main.APP_VERSION)
            self.assertEqual(report["frames"], {"total": 12, "accepted": 9, "skipped": 3})
            self.assertEqual(report["crop"]["width"], 400)
            self.assertEqual(report["stack"]["method"], "local")


if __name__ == "__main__":
    unittest.main()
