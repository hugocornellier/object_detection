part of '../../object_detection.dart';

/// Data passed to the detection isolate during startup.
class _DetectionIsolateStartupData {
  final SendPort sendPort;
  final TransferableTypedData modelBytes;
  final TransferableTypedData labelsBytes;
  final String modelName;
  final String performanceModeName;
  final int? numThreads;

  _DetectionIsolateStartupData({
    required this.sendPort,
    required this.modelBytes,
    required this.labelsBytes,
    required this.modelName,
    required this.performanceModeName,
    required this.numThreads,
  });
}

/// Direct-mode TFLite inference core used inside the detection background isolate.
///
/// Holds the loaded TFLite interpreter and label map, runs object detection
/// entirely on the calling thread (no further isolate spawning). Created
/// inside [ObjectDetector]'s background isolate by
/// [ObjectDetector._detectionIsolateEntry].
class _ObjectDetectorCore {
  ObjectDetection? _model;
  List<String> _labels = const <String>[];

  /// Returns true once initialized with model data.
  bool get isReady => _model != null;

  /// Initializes the model and label map from pre-loaded bytes.
  Future<void> initializeFromBuffers({
    required Uint8List modelBytes,
    required Uint8List labelsBytes,
    required ObjectDetectionModel model,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
  }) async {
    try {
      _model = await ObjectDetection.createFromBuffer(
        modelBytes,
        model,
        performanceConfig: performanceConfig,
      );
      final String labelText = utf8.decode(labelsBytes, allowMalformed: true);
      _labels = parseLabelMap(labelText);
    } catch (e) {
      _cleanupOnInitError();
      rethrow;
    }
  }

  /// Runs object detection directly on the calling thread.
  Future<List<DetectedObject>> detectDirect(
    cv.Mat image,
    ObjectDetectorOptions options,
  ) async {
    final m = _model;
    if (m == null) {
      throw StateError(
        'ObjectDetectorCore not initialized. Call initializeFromBuffers().',
      );
    }

    final int width = image.cols;
    final int height = image.rows;
    final Size imgSize = Size(width.toDouble(), height.toDouble());

    final tensor = convertImageToTensor(
      image,
      outW: m.inputWidth,
      outH: m.inputHeight,
    );
    // Pass the score threshold down to the decoder so we can short-circuit
    // sigmoid filtering on the per-anchor inner loop.
    final List<Detection> rawDets = await m.callWithTensor(
      tensor,
      scoreThreshold: options.scoreThreshold,
    );

    final filtered = _applyOptions(rawDets, options, _labels);

    final List<DetectedObject> results = <DetectedObject>[];
    for (final d in filtered) {
      final detWithSize = Detection(
        boundingBox: d.boundingBox,
        score: d.score,
        classIndex: d.classIndex,
        imageSize: imgSize,
      );
      final String name = d.classIndex >= 0 && d.classIndex < _labels.length
          ? _labels[d.classIndex]
          : '???';
      results.add(
        DetectedObject(
          detection: detWithSize,
          categories: [
            Category(
              index: d.classIndex,
              score: d.score,
              categoryName: name,
              displayName: name,
            ),
          ],
          originalSize: imgSize,
        ),
      );
    }

    return results;
  }

  /// Disposes the loaded model.
  void dispose() => _disposeFields();

  void _disposeFields({bool safe = false}) {
    void d(void Function() fn) {
      if (safe) {
        try {
          fn();
        } on StateError catch (_) {}
      } else {
        fn();
      }
    }

    d(() => _model?.dispose());
    _model = null;
  }

  void _cleanupOnInitError() => _disposeFields(safe: true);
}
