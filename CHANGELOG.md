## 0.4.0

* **Default precision is now `Precision.fp32` instead of `fp16`.** This changes
  numeric output. `flutter_litert` 3.8.0 changed its own default for the same
  reason: across 29 published detection models measured on five GPUs, fp16
  matched a plain-CPU reference for only about a fifth of them, while fp32
  matched every model that compiled. These graphs emit pixel-space coordinates
  and landmark positions, and fp16 carries about three decimal digits of
  mantissa, so the error lands directly on output geometry. The cost is real and
  worth stating plainly: fp32 is a median 29.9% slower on GPU across those five
  GPUs, with Apple M4 the lone exception at 6.5% faster. Pass
  `precision: Precision.fp16` explicitly to restore the previous behaviour,
  ideally per model and validated on your target GPU.
* Pin `flutter_litert` to `^3.8.0`.
* Add the LiteRT Next `CompiledModel` engine as an opt-in alternative to the
  `Interpreter` path, matching `face_detection_tflite`, `pose_detection` and
  `hand_detection`. `ObjectDetector.create` / `initialize` and
  `ObjectDetection.createCompiledFromBuffer` take `useCompiledModel`,
  `accelerators` and `precision`. Default remains the `Interpreter` path.
  Measured end-to-end on macOS (Apple Silicon): Lite0 1.3-2.1x faster, Lite2
  2.8-3.3x faster.
* The compiled engine uses `TensorBufferMode.hostMemory` and the
  `writeInput` / `dispatch` / `readOutput` zero-copy path, so the detection
  heads (~1.8M floats per frame for Lite0, ~3.4M for Lite2) are decoded as
  views of model-owned memory instead of being copied into fresh Dart lists.
  Measured 1.4x faster than the managed-buffer `runAsync` path. Falls back to
  managed buffers where host memory is unavailable.
* Preprocessing now uses OpenCV's SIMD `cvtColor` + `convertTo` kernels rather
  than a per-pixel Dart loop, via the new `bgrMatToSignedFloat32`. Measured
  4.1-4.5x faster at the tensor-conversion step, numerically equivalent to the
  scalar path (max absolute difference of one float32 ULP).
* `convertImageToTensor` skips `cv.resize` when the source already matches the
  target geometry and skips `cv.copyMakeBorder` when the letterbox is empty.
* The detection isolate reuses one input tensor buffer for its whole life
  instead of allocating 1.2 MB (Lite0) or 2.4 MB (Lite2) per frame.
  `ObjectDetection.newInputBuffer()` exposes the same for direct API users.
* Anchors are generated into a flat `Float32List` instead of one `List<double>`
  per anchor (19 206 of them for Lite0, 37 629 for Lite2), and the decode loop
  seeds its argmax at the score threshold and writes survivors into reusable
  typed scratch buffers. Measured 1.2x faster at the decode step, with
  identical output. New `generateEfficientDetAnchorsFlat`;
  `generateEfficientDetAnchors` is unchanged and now delegates to it.
* Add `ObjectDetection.usesCompiledModel` and
  `ObjectDetection.activeAccelerators` so callers can see which engine and
  accelerators a model actually compiled to.
* Re-export `Accelerator` and `Precision` from `flutter_litert`.
* Add a benchmark suite to the example: end-to-end and per-stage timings,
  invoke-vs-decode-vs-NMS attribution, a delegate/engine sweep, and an
  engine A/B that asserts the two engines agree. Each emits `BENCH_JSON`.
* The example app now defaults to the `CompiledModel` engine and carries a
  `CM` / `Interpreter` badge on all three screens that switches engines live,
  matching the face, pose and hand demos. Detector creation falls back to the
  `Interpreter` engine if `CompiledModel` cannot be created on the device, and
  the badge reports the engine actually in use. The package default is
  unchanged, so adding `object_detection` to an app never silently switches
  its inference engine.
* Warm the detector with one throwaway inference on creation. Without it the
  first timing shown after an engine switch is a cold `CompiledModel` number
  (Metal shader compilation) measured against a warm `Interpreter` one, which
  made the faster engine read as several times slower. Observed on Lite2:
  140 ms cold vs 39 ms warmed, against 41 ms for the interpreter.
* The badge names the engine (`CM` / `Interpreter`) rather than a delegate.
  The other demos label this axis `CM` / `XNN`, but XNNPACK is only the
  interpreter path's delegate on desktop and Android; on iOS it is Metal, so
  an `XNN` label is wrong there.

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
