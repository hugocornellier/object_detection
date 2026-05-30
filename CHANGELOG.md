## 0.1.0

* Update flutter_litert -> 2.8.0
* Complete Swift Package Manager migration: example apps build via SPM without CocoaPods

## 0.0.8

* Remove unused Darwin podspecs for Dart-only iOS/macOS plugin registration.

## 0.0.7

* Update flutter_litert -> 2.5.8
 
## 0.0.6

* Update flutter_litert -> 2.5.5

## 0.0.5

* Update flutter_litert to 2.5.3

# 0.0.4

* Update flutter_litert -> 2.5.2

# 0.0.3

* Update flutter_litert -> 2.5.0

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
