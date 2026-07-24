## 0.3.0

* Fix confidence decoding for both bundled EfficientDet-Lite variants. Their
  class tensors are already produced by a TFLite `LOGISTIC` op; scores and
  thresholds are now used directly instead of applying sigmoid a second time.
  This removes widespread false positives and restores calibrated confidence
  values.
* Bump `ObjectDetector.modelVersion` because postprocessing output changes for
  the same input bytes.
* Add real-model regressions for Lite0 and Lite2 blank/uniform inputs.
* **Breaking:** trim the `flutter_litert` convenience re-export surface. Removed
  16 symbols that this package never used and never documented: `createNHWCTensor4D`,
  `fillNHWC4D`, `allocTensorShape`, `flattenDynamicTensor`, `sigmoid`,
  `sigmoidClipped`, `bgrBytesToRgbFloat32`, `packYuv420`, `YuvPlane`,
  `YuvLayout`, `PackedYuv`, `CameraPlane`, `coverFitScaleOffset`,
  `drawLandmarkMarker`, `drawSkeletonConnections`, `drawBoundingBoxOutline`. If
  you relied on any of these, import them directly from
  `package:flutter_litert/flutter_litert.dart`.
* Deprecate `OutputTensorInfo`, `collectOutputTensorInfo` and
  `testCollectOutputTensorInfo`. They are byte-identical to copies in
  `face_detection_tflite` and are not called anywhere in this package outside
  their own test hook; `flutter_litert` 3.6.0 provides `collectOutputShapes`,
  which returns shapes without materializing tensor buffers. Deprecated rather
  than removed, since they are public via `part`.
* Update flutter_litert -> 3.6.0.
* Expand the README live camera section with the full production pipeline
  (frame throttling, orientation handling, cover-fit overlay mapping).

## 0.2.3

* Update flutter_litert -> 3.5.0

## 0.2.2

* Update flutter_litert -> 3.4.1 (web `CompiledModel` WebGPU compile watchdog: a compile attempt that never settles now falls back to WASM instead of hanging). No API change.

## 0.2.1

* Update flutter_litert -> 3.3.1

## 0.2.0

* Update flutter_litert -> 3.2.0
* Import native-only flutter_litert APIs via `package:flutter_litert/native.dart` so they resolve under static analysis (flutter_litert 3.2.0 moved `InterpreterFactory`, `IsolateWorkerBase`, and `TensorFloat32Views` behind the native conditional export). No runtime or API change.

## 0.1.2

* Update flutter_litert -> 3.1.1

## 0.1.1

* Update flutter_litert -> 2.8.3

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
