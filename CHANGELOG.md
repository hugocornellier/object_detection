# 0.0.2

* Update flutter_litert -> 2.4.1

# 0.0.1 (2026-04-27)

* Initial release.
* On-device object detection over 80 COCO classes.
* Two model variants: EfficientDet-Lite0 (default, 320×320) and EfficientDet-Lite2 (448×448).
* Per-call options: `scoreThreshold`, `maxResults`, `categoryAllowlist`, `categoryDenylist`.
* Background isolate for inference; UI thread is never blocked.
* Image input variants: encoded bytes, file path, `cv.Mat`, raw pixel bytes, `CameraImage`, `CameraFrame`.
* Hardware-accelerated by default: Metal on iOS, XNNPACK elsewhere.
* Cross-platform: Android, iOS, macOS, Windows, Linux.
