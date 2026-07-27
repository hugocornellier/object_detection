part of '../../object_detection.dart';

/// Data passed to the detection isolate during startup.
class _DetectionIsolateStartupData {
  final SendPort sendPort;
  final TransferableTypedData modelBytes;
  final TransferableTypedData labelsBytes;
  final String modelName;
  final String performanceModeName;
  final int? numThreads;
  final bool useCompiledModel;

  /// [Accelerator] and [Precision] enum indices rather than the enums
  /// themselves, so the startup payload stays plain data across the isolate
  /// boundary.
  final List<int> acceleratorIndices;
  final int precisionIndex;

  _DetectionIsolateStartupData({
    required this.sendPort,
    required this.modelBytes,
    required this.labelsBytes,
    required this.modelName,
    required this.performanceModeName,
    required this.numThreads,
    required this.useCompiledModel,
    required this.acceleratorIndices,
    required this.precisionIndex,
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

  /// Input tensor reused across detections. The core is pinned to one model for
  /// its whole life, so the buffer is allocated once at init and every frame
  /// writes over it instead of allocating `inW * inH * 3` floats (1.2 MB for
  /// Lite0, 2.4 MB for Lite2) per call.
  Float32List? _inputBuffer;

  /// Returns true once initialized with model data.
  bool get isReady => _model != null;

  /// The accelerators the CompiledModel engine compiled to, or null when the
  /// interpreter path is in use.
  Set<Accelerator>? get activeAccelerators => _model?.activeAccelerators;

  /// Initializes the model and label map from pre-loaded bytes.
  ///
  /// When [useCompiledModel] is true the LiteRT Next [CompiledModel] engine
  /// backs inference and [performanceConfig] is ignored (that knob only
  /// configures [Interpreter] delegates).
  Future<void> initializeFromBuffers({
    required Uint8List modelBytes,
    required Uint8List labelsBytes,
    required ObjectDetectionModel model,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    bool useCompiledModel = false,
    Set<Accelerator> accelerators = const {Accelerator.gpu, Accelerator.cpu},
    Precision precision = Precision.fp16,
  }) async {
    try {
      _model = useCompiledModel
          ? await ObjectDetection.createCompiledFromBuffer(
              modelBytes,
              model,
              accelerators: accelerators,
              precision: precision,
            )
          : await ObjectDetection.createFromBuffer(
              modelBytes,
              model,
              performanceConfig: performanceConfig,
            );
      _inputBuffer = _model!.newInputBuffer();
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
      buffer: _inputBuffer,
    );
    // Pass the score threshold to the decoder so sub-threshold anchors are
    // dropped during decoding, before NMS.
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
    _inputBuffer = null;
  }

  void _cleanupOnInitError() => _disposeFields(safe: true);
}
