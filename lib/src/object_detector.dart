part of '../object_detection.dart';

/// On-device object detection using MediaPipe TFLite models.
///
/// Wraps a single TFLite object-detection model and runs it inside a background
/// isolate so the UI thread is never blocked during inference.
///
/// ## Usage
///
/// ```dart
/// // One-step construction
/// final detector = await ObjectDetector.create();
///
/// // Or two-step, if you need to configure between construction and init
/// final detector = ObjectDetector();
/// await detector.initialize();
///
/// // Detect objects with default options (score threshold 0.5)
/// final detections = await detector.detect(imageBytes);
///
/// // Or customize per call
/// final detections = await detector.detect(
///   imageBytes,
///   options: const ObjectDetectorOptions(scoreThreshold: 0.3, maxResults: 5),
/// );
///
/// for (final obj in detections) {
///   print('${obj.categoryName}: ${obj.score.toStringAsFixed(2)}');
/// }
///
/// // Clean up when done
/// await detector.dispose();
/// ```
///
/// ## Lifecycle
///
/// 1. Create instance with `ObjectDetector()`
/// 2. Call [initialize] to load the TFLite model
/// 3. Check [isReady] to verify the model is loaded
/// 4. Call [detect] to analyze images
/// 5. Call [dispose] when done to free resources
class ObjectDetector {
  /// Cache-invalidation key for consumers that persist detection results.
  ///
  /// Bump this when detection output could change for the same input bytes.
  /// For example, on model file swaps, threshold changes, preprocessing changes,
  /// or postprocessing / coordinate-space changes.
  static const String modelVersion = '1.0.0';

  /// Creates a new object detector instance.
  ///
  /// The detector is not ready for use until you call [initialize].
  ObjectDetector();

  /// Creates and initializes an object detector in one step.
  ///
  /// Convenience factory that combines [ObjectDetector.new] and [initialize].
  /// Accepts the same parameters as [initialize].
  ///
  /// Example:
  /// ```dart
  /// final detector = await ObjectDetector.create();
  ///
  /// // Equivalent to:
  /// final detector = ObjectDetector();
  /// await detector.initialize();
  /// ```
  static Future<ObjectDetector> create({
    ObjectDetectionModel model = ObjectDetectionModel.efficientDetLite0,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
  }) async {
    final detector = ObjectDetector();
    await detector.initialize(
      model: model,
      performanceConfig: performanceConfig,
    );
    return detector;
  }

  _ObjectDetectorWorker? _worker;

  /// Returns true if the model is loaded and ready for inference.
  bool get isReady => _worker?.isReady ?? false;

  /// Loads the object detection model and prepares the interpreter for
  /// inference in a background isolate.
  ///
  /// Must be called before running any detections.
  /// Calling [initialize] twice without [dispose] throws [StateError].
  ///
  /// The [model] argument specifies which TFLite model variant to load.
  /// The [performanceConfig] parameter controls hardware acceleration:
  /// - iOS: Metal GPU delegate (auto)
  /// - Android/macOS/Linux/Windows: XNNPACK (auto)
  Future<void> initialize({
    ObjectDetectionModel model = ObjectDetectionModel.efficientDetLite0,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
  }) async {
    if (isReady) {
      throw StateError('ObjectDetector already initialized');
    }

    final worker = _ObjectDetectorWorker();

    try {
      final modelPath =
          'packages/object_detection/assets/models/${_nameFor(model)}';
      final labelsPath =
          'packages/object_detection/assets/models/$_labelMapAsset';

      final results = await Future.wait([
        rootBundle.load(modelPath),
        rootBundle.load(labelsPath),
      ]);

      await worker.initialize(
        modelBytes: results[0].buffer.asUint8List(),
        labelsBytes: results[1].buffer.asUint8List(),
        model: model,
        performanceConfig: performanceConfig,
      );

      _worker = worker;
    } catch (_) {
      if (worker.isReady) {
        await worker.dispose();
      }
      rethrow;
    }
  }

  /// Detects objects in encoded image bytes.
  ///
  /// The [imageBytes] parameter should contain encoded image data
  /// (JPEG, PNG, etc.). For pre-decoded [cv.Mat] input, use
  /// [detectFromMat] instead.
  ///
  /// Returns a [List] of [DetectedObject] sorted by descending score.
  ///
  /// Throws [StateError] if [initialize] has not been called successfully.
  /// Throws [FormatException] if the image bytes cannot be decoded.
  Future<List<DetectedObject>> detect(
    Uint8List imageBytes, {
    ObjectDetectorOptions options = ObjectDetectorOptions.defaults,
  }) async {
    _requireReady();
    _validateOptions(options);
    final List<dynamic> result = await _sendDetectionRequest<List<dynamic>>(
      'detect',
      {
        'bytes': TransferableTypedData.fromList([imageBytes]),
        'options': options.toMap(),
      },
    );
    return _deserializeResults(result);
  }

  static void _validateOptions(ObjectDetectorOptions options) {
    if (options.categoryAllowlist.isNotEmpty &&
        options.categoryDenylist.isNotEmpty) {
      throw ArgumentError(
        'categoryAllowlist and categoryDenylist are mutually exclusive. '
        'Pass at most one.',
      );
    }
  }

  /// Detects objects in an image file at [path].
  ///
  /// Convenience wrapper that reads the file and calls [detect].
  /// Not available on Flutter Web (uses `dart:io`).
  Future<List<DetectedObject>> detectFromFilepath(
    String path, {
    ObjectDetectorOptions options = ObjectDetectorOptions.defaults,
  }) async {
    final bytes = await File(path).readAsBytes();
    return detect(bytes, options: options);
  }

  /// Detects objects in a pre-decoded [cv.Mat] image.
  ///
  /// The Mat is NOT disposed by this method. The caller is responsible for disposal.
  Future<List<DetectedObject>> detectFromMat(
    cv.Mat image, {
    ObjectDetectorOptions options = ObjectDetectorOptions.defaults,
  }) {
    _requireReady();
    final f = _extractMatFields(image);
    return detectFromMatBytes(
      f.data,
      width: f.width,
      height: f.height,
      matType: f.matType,
      options: options,
    );
  }

  /// Detects objects from raw pixel bytes without constructing a [cv.Mat] first.
  ///
  /// The bytes are transferred via zero-copy [TransferableTypedData] and the
  /// Mat is reconstructed inside the background isolate.
  ///
  /// Parameters:
  /// - [bytes]: Raw pixel data (e.g. BGR, BGRA)
  /// - [width]: Image width in pixels
  /// - [height]: Image height in pixels
  /// - [matType]: OpenCV MatType value (default: CV_8UC3 = 16 for BGR)
  Future<List<DetectedObject>> detectFromMatBytes(
    Uint8List bytes, {
    required int width,
    required int height,
    int matType = 16,
    ObjectDetectorOptions options = ObjectDetectorOptions.defaults,
  }) async {
    _requireReady();
    _validateOptions(options);
    final List<dynamic> result = await _sendDetectionRequest<List<dynamic>>(
      'detectMat',
      {
        'bytes': TransferableTypedData.fromList([bytes]),
        'width': width,
        'height': height,
        'matType': matType,
        'options': options.toMap(),
      },
    );
    return _deserializeResults(result);
  }

  /// Detects objects directly from a [CameraFrame] produced by
  /// [prepareCameraFrame].
  ///
  /// The frame's YUV→BGR colour conversion and any optional rotation happen
  /// inside the detection isolate.
  Future<List<DetectedObject>> detectFromCameraFrame(
    CameraFrame frame, {
    ObjectDetectorOptions options = ObjectDetectorOptions.defaults,
    int? maxDim,
  }) async {
    _requireReady();
    _validateOptions(options);
    final List<dynamic> result = await _sendDetectionRequest<List<dynamic>>(
      'detectCameraFrame',
      _cameraFrameFields(frame, {
        'options': options.toMap(),
        'maxDim': maxDim,
      }),
    );
    return _deserializeResults(result);
  }

  /// One-call wrapper for live camera streams.
  ///
  /// Takes a `CameraImage`-shaped object directly (any object exposing
  /// `width`, `height`, and `planes` with `bytes` / `bytesPerRow` /
  /// `bytesPerPixel`) and runs YUV packing, colour conversion, rotation,
  /// and downscale in the detection isolate, all off the UI thread.
  Future<List<DetectedObject>> detectFromCameraImage(
    Object cameraImage, {
    ObjectDetectorOptions options = ObjectDetectorOptions.defaults,
    CameraFrameRotation? rotation,
    bool isBgra = true,
    int? maxDim,
  }) async {
    _requireReady();
    final frame = prepareCameraFrameFromImage(
      cameraImage,
      rotation: rotation,
      isBgra: isBgra,
    );
    if (frame == null) return const <DetectedObject>[];
    return detectFromCameraFrame(frame, options: options, maxDim: maxDim);
  }

  /// Releases all resources held by the detector.
  ///
  /// After calling dispose, you must call [initialize] again before
  /// running any detections.
  Future<void> dispose() async {
    final worker = _worker;
    _worker = null;
    if (worker != null && worker.isReady) {
      await worker.dispose();
    }
  }

  void _requireReady() {
    if (!isReady) {
      throw StateError(
        'ObjectDetector not initialized. Call initialize() before using.',
      );
    }
  }

  Future<T> _sendDetectionRequest<T>(
    String operation,
    Map<String, dynamic> params,
  ) =>
      _worker!.sendRequest<T>(operation, params);

  static List<DetectedObject> _deserializeResults(List<dynamic> result) =>
      result
          .map((m) =>
              DetectedObject.fromMap(Map<String, dynamic>.from(m as Map)))
          .toList();

  static Uint8List _extractBytes(dynamic message) =>
      (message['bytes'] as TransferableTypedData).materialize().asUint8List();

  static cv.Mat _matFromMessage(Map message, Uint8List bytes) {
    final int width = message['width'] as int;
    final int height = message['height'] as int;
    final int matTypeValue = message['matType'] as int;
    return cv.Mat.fromList(height, width, cv.MatType(matTypeValue), bytes);
  }

  /// Builds the isolate-request field map for a [CameraFrame] payload, merged
  /// with any [extra] per-op fields.
  Map<String, dynamic> _cameraFrameFields(
    CameraFrame frame,
    Map<String, dynamic> extra,
  ) =>
      {
        'bytes': TransferableTypedData.fromList([frame.bytes]),
        'width': frame.width,
        'height': frame.height,
        'strideCols': frame.strideCols,
        'conversion': frame.conversion.index,
        'rotation': frame.rotation?.index,
        ...extra,
      };

  /// Decodes a [CameraFrame] message into a 3-channel BGR [cv.Mat] inside the
  /// detection isolate. Op ordering is tuned to keep big buffers small.
  static cv.Mat _matFromCameraFrameMessage(Map message, Uint8List bytes) {
    final int width = message['width'] as int;
    final int height = message['height'] as int;
    final int strideCols = message['strideCols'] as int;
    final conversion =
        CameraFrameConversion.values[message['conversion'] as int];
    final int? rotationIndex = message['rotation'] as int?;
    final int? maxDim = message['maxDim'] as int?;

    int? rotateFlag() {
      if (rotationIndex == null) return null;
      return switch (CameraFrameRotation.values[rotationIndex]) {
        CameraFrameRotation.cw90 => cv.ROTATE_90_CLOCKWISE,
        CameraFrameRotation.cw180 => cv.ROTATE_180,
        CameraFrameRotation.cw270 => cv.ROTATE_90_COUNTERCLOCKWISE,
      };
    }

    cv.Mat maybeResize(cv.Mat m) {
      if (maxDim == null || (m.cols <= maxDim && m.rows <= maxDim)) return m;
      final double scale = maxDim / (m.cols > m.rows ? m.cols : m.rows);
      final resized = cv.resize(
        m,
        ((m.cols * scale).toInt(), (m.rows * scale).toInt()),
        interpolation: cv.INTER_LINEAR,
      );
      m.dispose();
      return resized;
    }

    cv.Mat maybeRotate(cv.Mat m) {
      final flag = rotateFlag();
      if (flag == null) return m;
      final rotated = cv.rotate(m, flag);
      m.dispose();
      return rotated;
    }

    switch (conversion) {
      case CameraFrameConversion.bgra2bgr:
      case CameraFrameConversion.rgba2bgr:
        final bgraOrRgba =
            cv.Mat.fromList(height, strideCols, cv.MatType.CV_8UC4, bytes);
        cv.Mat current = strideCols != width
            ? bgraOrRgba.region(cv.Rect(0, 0, width, height))
            : bgraOrRgba;

        if (maxDim != null &&
            (current.cols > maxDim || current.rows > maxDim)) {
          final double scale = maxDim /
              (current.cols > current.rows ? current.cols : current.rows);
          final resized = cv.resize(
            current,
            ((current.cols * scale).toInt(), (current.rows * scale).toInt()),
            interpolation: cv.INTER_LINEAR,
          );
          if (!identical(current, bgraOrRgba)) current.dispose();
          current = resized;
        }

        final flag = rotateFlag();
        if (flag != null) {
          final rotated = cv.rotate(current, flag);
          if (!identical(current, bgraOrRgba)) current.dispose();
          current = rotated;
        }

        final cvtCode = conversion == CameraFrameConversion.bgra2bgr
            ? cv.COLOR_BGRA2BGR
            : cv.COLOR_RGBA2BGR;
        final bgr = cv.cvtColor(current, cvtCode);
        if (!identical(current, bgraOrRgba)) current.dispose();
        bgraOrRgba.dispose();
        return bgr;

      case CameraFrameConversion.yuv2bgrNv12:
      case CameraFrameConversion.yuv2bgrNv21:
      case CameraFrameConversion.yuv2bgrI420:
        final yuvMat = cv.Mat.fromList(
          height + height ~/ 2,
          width,
          cv.MatType.CV_8UC1,
          bytes,
        );
        final cvtCode = switch (conversion) {
          CameraFrameConversion.yuv2bgrNv12 => cv.COLOR_YUV2BGR_NV12,
          CameraFrameConversion.yuv2bgrNv21 => cv.COLOR_YUV2BGR_NV21,
          CameraFrameConversion.yuv2bgrI420 => cv.COLOR_YUV2BGR_I420,
          _ => cv.COLOR_YUV2BGR_NV12,
        };
        cv.Mat current = cv.cvtColor(yuvMat, cvtCode);
        yuvMat.dispose();
        current = maybeResize(current);
        current = maybeRotate(current);
        return current;
    }
  }

  ({Uint8List data, int width, int height, int matType}) _extractMatFields(
    cv.Mat image,
  ) =>
      (
        data: image.data,
        width: image.cols,
        height: image.rows,
        matType: image.type.value,
      );

  /// Detection isolate entry point.
  @pragma('vm:entry-point')
  static void _detectionIsolateEntry(_DetectionIsolateStartupData data) async {
    final SendPort mainSendPort = data.sendPort;
    final ReceivePort workerReceivePort = ReceivePort();

    _ObjectDetectorCore? core;

    try {
      final modelBytes = data.modelBytes.materialize().asUint8List();
      final labelsBytes = data.labelsBytes.materialize().asUint8List();

      final model = ObjectDetectionModel.values.firstWhere(
        (m) => m.name == data.modelName,
      );
      final performanceMode = PerformanceMode.values.firstWhere(
        (m) => m.name == data.performanceModeName,
      );

      core = _ObjectDetectorCore();
      await core.initializeFromBuffers(
        modelBytes: modelBytes,
        labelsBytes: labelsBytes,
        model: model,
        performanceConfig: PerformanceConfig(
          mode: performanceMode,
          numThreads: data.numThreads,
        ),
      );

      mainSendPort.send(workerReceivePort.sendPort);
    } catch (e, st) {
      mainSendPort.send({
        'error': 'Detection isolate initialization failed: $e\n$st',
      });
      return;
    }

    workerReceivePort.listen((message) async {
      if (message is! Map) return;

      final int? id = message['id'] as int?;
      final String? op = message['op'] as String?;

      if (id == null || op == null) return;

      try {
        switch (op) {
          case 'detect':
            if (core == null) {
              mainSendPort.send({
                'id': id,
                'error': 'ObjectDetectorCore not initialized in isolate',
              });
              return;
            }
            final Uint8List imageBytes = _extractBytes(message);
            final options = ObjectDetectorOptions.fromMap(
              Map<String, dynamic>.from(message['options'] as Map),
            );
            final cv.Mat mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
            try {
              final dets = await core!.detectDirect(mat, options);
              mainSendPort.send({
                'id': id,
                'result': dets.map((d) => d.toMap()).toList(),
              });
            } finally {
              mat.dispose();
            }

          case 'detectMat':
            if (core == null) {
              mainSendPort.send({
                'id': id,
                'error': 'ObjectDetectorCore not initialized in isolate',
              });
              return;
            }
            final Uint8List matBytes = _extractBytes(message);
            final options = ObjectDetectorOptions.fromMap(
              Map<String, dynamic>.from(message['options'] as Map),
            );
            final mat = _matFromMessage(message, matBytes);
            try {
              final dets = await core!.detectDirect(mat, options);
              mainSendPort.send({
                'id': id,
                'result': dets.map((d) => d.toMap()).toList(),
              });
            } finally {
              mat.dispose();
            }

          case 'detectCameraFrame':
            if (core == null) {
              mainSendPort.send({
                'id': id,
                'error': 'ObjectDetectorCore not initialized in isolate',
              });
              return;
            }
            final Uint8List frameBytes = _extractBytes(message);
            final options = ObjectDetectorOptions.fromMap(
              Map<String, dynamic>.from(message['options'] as Map),
            );
            final frameMat = _matFromCameraFrameMessage(message, frameBytes);
            try {
              final dets = await core!.detectDirect(frameMat, options);
              mainSendPort.send({
                'id': id,
                'result': dets.map((d) => d.toMap()).toList(),
              });
            } finally {
              frameMat.dispose();
            }

          case 'dispose':
            core?.dispose();
            core = null;
            workerReceivePort.close();
        }
      } catch (e, st) {
        mainSendPort.send({'id': id, 'error': '$e\n$st'});
      }
    });
  }
}

class _ObjectDetectorWorker extends IsolateWorkerBase {
  @override
  String get workerDisposeOp => 'dispose';

  Future<void> initialize({
    required Uint8List modelBytes,
    required Uint8List labelsBytes,
    required ObjectDetectionModel model,
    required PerformanceConfig performanceConfig,
  }) async {
    await initWorker(
      (sendPort) => Isolate.spawn(
        ObjectDetector._detectionIsolateEntry,
        _DetectionIsolateStartupData(
          sendPort: sendPort,
          modelBytes: TransferableTypedData.fromList([modelBytes]),
          labelsBytes: TransferableTypedData.fromList([labelsBytes]),
          modelName: model.name,
          performanceModeName: performanceConfig.mode.name,
          numThreads: performanceConfig.numThreads,
        ),
        debugName: 'ObjectDetector.detection',
      ),
      timeout: const Duration(seconds: 30),
      timeoutMessage: 'Detection isolate initialization timed out',
    );
  }
}
