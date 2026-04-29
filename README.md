<h1 align="center">object_detection</h1>

<p align="center">
<a href="https://flutter.dev"><img src="https://img.shields.io/badge/Platform-Flutter-02569B?logo=flutter" alt="Platform"></a>
<a href="https://dart.dev"><img src="https://img.shields.io/badge/language-Dart-blue" alt="Language: Dart"></a>
<br>
<a href="https://pub.dev/packages/object_detection"><img src="https://img.shields.io/pub/v/object_detection?label=pub.dev&labelColor=333940&logo=dart" alt="Pub Version"></a>
<a href="https://github.com/hugocornellier/object_detection/actions/workflows/build.yml"><img src="https://github.com/hugocornellier/object_detection/actions/workflows/build.yml/badge.svg" alt="CI"></a>
<a href="https://github.com/hugocornellier/object_detection/actions/workflows/integration.yml"><img src="https://github.com/hugocornellier/object_detection/actions/workflows/integration.yml/badge.svg" alt="Tests"></a>
<a href="https://github.com/hugocornellier/object_detection/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-007A88.svg?logo=apache" alt="License"></a>
</p>

Flutter implementation of Google's [MediaPipe Object Detector](https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector) using LiteRT (formerly TensorFlow Lite). Detects 80 COCO object classes (person, car, cat, dog, ...) with bounding boxes and confidence scores. Completely local: no remote API, just pure on-device, offline detection.

## Features

- On-device object detection across 80 COCO classes, runs fully offline
- Bounding boxes + class labels + confidence scores
- Two model variants tradeable for accuracy vs. speed (EfficientDet-Lite0, EfficientDet-Lite2)
- Per-call options: score threshold, max results, category allow / deny lists
- Truly cross-platform: compatible with Android, iOS, macOS, Windows, and Linux
- Background isolate: the UI thread is never blocked during inference
- Live camera support: YUV/BGRA conversion + rotation + downscale all run off the UI thread

## Quick Start

```dart
import 'package:object_detection/object_detection.dart';

Future main() async {
  // Initialize detector, run inference on image
  ObjectDetector detector = await ObjectDetector.create();
  List<DetectedObject> detections = await detector.detect(imageBytes);

  // Iterate through detected objects
  for (final obj in detections) {
    print('${obj.categoryName} (${(obj.score * 100).toStringAsFixed(1)}%) '
          'at (${obj.boundingBox.topLeft.x.toInt()}, '
          '${obj.boundingBox.topLeft.y.toInt()})');
  }

  await detector.dispose();
}
```

Already have bytes (from a file or the network)? Use `detect(imageBytes)`. For live camera streams, use `detectFromCameraImage(...)` (keeps all OpenCV work off the UI thread, see below). For a pre-decoded `cv.Mat`, use `detectFromMat(mat)`.

## Models

All TFLite models are sourced from Google's [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector) framework:

| Model | File | Input | Best For |
|-------|------|-------|----------|
| EfficientDet-Lite0 (default) | `efficientdet_lite0.tflite` | 320×320 | Balanced speed/accuracy |
| EfficientDet-Lite2 | `efficientdet_lite2.tflite` | 448×448 | Higher accuracy, slower |

Both models output detections over 80 COCO classes (90 entries in the label
map; some are placeholder `???` slots to keep alignment with the original
COCO category IDs).

## Bounding Boxes

The `boundingBox` property returns a `BoundingBox` representing the object bounding box in absolute pixel coordinates. The `BoundingBox` provides convenient access to corner points, dimensions, and center.

### Accessing Corners

```dart
final BoundingBox boundingBox = obj.boundingBox;

// Access individual corners by name (each is a Point with x and y)
final Point topLeft     = boundingBox.topLeft;
final Point topRight    = boundingBox.topRight;
final Point bottomRight = boundingBox.bottomRight;
final Point bottomLeft  = boundingBox.bottomLeft;

print('Top-left: (${topLeft.x}, ${topLeft.y})');
```

### Additional Bounding Box Parameters

```dart
final BoundingBox boundingBox = obj.boundingBox;

final double width  = boundingBox.width;
final double height = boundingBox.height;
final Point center  = boundingBox.center;

print('Size: ${width} x ${height}');
print('Center: (${center.x}, ${center.y})');

// All corners as a list (order: top-left, top-right, bottom-right, bottom-left)
final List<Point> allCorners = boundingBox.corners;
```

## Categories

Each detection carries one or more `Category` objects with the predicted class
index, score, and label string. For object detection the model emits one top
class per box, so `obj.category` gives the dominant class:

```dart
final cat = obj.category;
print('${cat.categoryName} (index ${cat.index}, score ${cat.score})');
```

## Per-call Options

`detect(...)` accepts an optional `ObjectDetectorOptions` for filtering:

```dart
// Threshold + cap
final results = await detector.detect(
  imageBytes,
  options: const ObjectDetectorOptions(
    scoreThreshold: 0.5,
    maxResults: 5,
  ),
);

// Only people and cars
final filtered = await detector.detect(
  imageBytes,
  options: const ObjectDetectorOptions(
    scoreThreshold: 0.4,
    categoryAllowlist: ['person', 'car'],
  ),
);

// Or exclude certain classes
final hideTraffic = await detector.detect(
  imageBytes,
  options: const ObjectDetectorOptions(
    categoryDenylist: ['traffic light', 'stop sign'],
  ),
);
```

`categoryAllowlist` and `categoryDenylist` are mutually exclusive. Pass at
most one.

## Live Camera Detection

For real-time object detection with a camera feed, use `detectFromCameraImage`. It auto-detects YUV420 (NV12 / NV21 / I420) and desktop BGRA/RGBA layouts, and the `cvtColor`, optional `rotate`, and `maxDim` downscale all run inside the detector's existing isolate: the UI thread is never blocked by OpenCV work.

```dart
import 'package:camera/camera.dart';
import 'package:object_detection/object_detection.dart';

final detector = await ObjectDetector.create();

final cameras = await availableCameras();
final camera = CameraController(
  cameras.first,
  ResolutionPreset.medium,
  enableAudio: false,
  imageFormatGroup: ImageFormatGroup.yuv420,
);
await camera.initialize();

camera.startImageStream((CameraImage image) async {
  final detections = await detector.detectFromCameraImage(
    image,
    // rotation: CameraFrameRotation.cw90, // based on device orientation
    options: const ObjectDetectorOptions(scoreThreshold: 0.5, maxResults: 10),
    maxDim: 640, // optional in-isolate downscale before inference
  );
  // Process detections...
});
```

**Tips for camera detection:**
- `detectFromCameraImage` replaces the old `packYuv420` + manual `cv.cvtColor` + `cv.rotate` dance in one call; no `cv.Mat` on the UI thread.
- Pass `rotation:` so the detector sees upright frames (Android back/front + device orientation logic); on iOS the camera plugin pre-rotates so this is often null.
- Pass `maxDim:` (e.g. 640) to downscale in-isolate; the detection model internally resizes to 320–448 px, so full-res frames just waste IPC bandwidth.
- Mirror the overlay on the front camera to match `CameraPreview`'s auto-mirrored texture.
- For advanced reuse, the underlying two-step API is `prepareCameraFrame(...)` + `detectFromCameraFrame(...)`.

## Background Processing

All inference runs automatically in a background isolate: the UI thread is never blocked during anchor decoding, NMS, or label resolution. No special configuration is needed; `ObjectDetector` handles isolate management internally.

## Performance

### Hardware Acceleration

The package automatically selects the best acceleration strategy for each platform:

| Platform | Default Delegate | Speedup | Notes |
|----------|-----------------|---------|-------|
| **macOS** | XNNPACK | 2-5x | SIMD vectorization (NEON on ARM, AVX on x86) |
| **Linux** | XNNPACK | 2-5x | SIMD vectorization |
| **iOS** | Metal GPU | 2-4x | Hardware GPU acceleration |
| **Android** | XNNPACK | 2-5x | ARM NEON SIMD acceleration |
| **Windows** | XNNPACK | 2-5x | SIMD vectorization (AVX on x86) |

No configuration needed: just call `ObjectDetector.create()` (or `initialize()`) and you get the optimal performance for your platform.

### Measured latency

Median per-image latency on `cat.jpg` (640×480) with EfficientDet-Lite0 after warm-up:

| Platform | Median |
|----------|--------|
| macOS host (Apple Silicon) | ~31 ms |
| iPhone 16 Pro simulator | ~40 ms |
| Android emulator (Pixel 8, API 36) | ~41 ms |

### Advanced Performance Configuration

The `performanceConfig` parameter works on both `create()` and `initialize()`.

```dart
// Auto mode (default): optimal for each platform
final detector = await ObjectDetector.create();
// Equivalent to:
final detector = await ObjectDetector.create(
  performanceConfig: PerformanceConfig.auto(),
);

// Force XNNPACK (all native platforms)
final detector = await ObjectDetector.create(
  performanceConfig: PerformanceConfig.xnnpack(numThreads: 4),
);

// Force GPU delegate (iOS recommended, Android experimental)
final detector = await ObjectDetector.create(
  performanceConfig: PerformanceConfig.gpu(),
);

// CPU-only (maximum compatibility)
final detector = await ObjectDetector.create(
  performanceConfig: PerformanceConfig.disabled,
);
```

### Advanced: Direct Mat Input

For live camera streams, you can bypass image encoding/decoding entirely by passing a `Mat` directly to `detectFromMat()`:

```dart
import 'package:object_detection/object_detection.dart';

Future<void> processFrame(Mat frame) async {
  final detector = await ObjectDetector.create();

  // Direct Mat input: fastest for video streams
  final detections = await detector.detectFromMat(frame);

  frame.dispose(); // always dispose Mats after use
  await detector.dispose();
}
```

**When to use `Mat` input:**
- You already have a decoded `cv.Mat` from another OpenCV pipeline
- You need to preprocess images with OpenCV before detection

For live camera streams, prefer `detectFromCameraImage(...)`: it keeps all `cvtColor` / `rotate` / downscale work inside the detection isolate rather than on the UI thread.

**For all other cases**, pass image bytes (`Uint8List`) to `detect()`.

### Advanced: Raw Pixel Bytes Input

If you already have raw pixel data as a `Uint8List` (e.g. from an isolate worker or image processing pipeline), use `detectFromMatBytes()` to skip constructing a `cv.Mat` on the calling thread entirely:

```dart
final Uint8List rawPixels = ...;
final int width = 1920;
final int height = 1080;

final detections = await detector.detectFromMatBytes(
  rawPixels,
  width: width,
  height: height,
  // matType: 16 (CV_8UC3/BGR) is the default
);
```

This is the fastest path when you already have raw pixel bytes: the data is transferred to the background isolate via zero-copy `TransferableTypedData`, and the `cv.Mat` is reconstructed there instead of on the calling thread.

### Memory Considerations

`ObjectDetector` holds the TFLite model (~14 MB for EfficientDet-Lite0, ~24 MB for Lite2) in a background isolate. Always call `dispose()` when finished to release these resources. Image data is transferred using zero-copy `TransferableTypedData`, minimizing memory overhead.

## Example

The [sample code](https://pub.dev/packages/object_detection/example) from the pub.dev example tab includes a Flutter app demonstrating:

- Bounding boxes with class labels and confidence scores
- Compare `efficientDetLite0` and `efficientDetLite2` models side-by-side
- Adjustable score threshold and max-results sliders
- Real-time inference timing display

## Inspiration

This package is built on top of Google's [MediaPipe Object Detector](https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector) models and structurally mirrors the sister plugin **[face_detection_tflite](https://pub.dev/packages/face_detection_tflite)**, sharing the same isolate-based architecture, performance configuration, camera frame handling, and OpenCV/LiteRT pipeline.
