part of '../../object_detection.dart';

/// Output binding for an EfficientDet detector.
///
/// Both EfficientDet TFLite outputs are 3D `[1, numAnchors, X]`:
///   - boxes: shape `[1, A, 4]`, `[ty, tx, th, tw]` deltas relative to anchor.
///   - classes: shape `[1, A, K]`, raw per-class logits (sigmoid in postproc).
class _DetectorOutputBinding {
  final int boxesIdx;
  final int classesIdx;
  final int numAnchors;
  final int numClasses;

  const _DetectorOutputBinding({
    required this.boxesIdx,
    required this.classesIdx,
    required this.numAnchors,
    required this.numClasses,
  });
}

/// Runs object detection on an input image and returns raw [Detection]
/// records (normalized coordinates).
///
/// The underlying TFLite models are sourced from Google's MediaPipe
/// Object Detector solution. See the model card and download links at:
/// https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector
///
/// These TFLite files emit raw RetinaNet-style anchor outputs; this class
/// generates the anchors at load time, then applies sigmoid + per-anchor
/// argmax + box decoding + weighted NMS in Dart to produce final detections.
///
/// Most users should use the high-level [ObjectDetector] class instead of
/// working with this low-level model API directly.
class ObjectDetection with _TfliteModelDisposable {
  @override
  final Interpreter _itp;
  final int _inW, _inH;
  late final _DetectorOutputBinding _binding;
  late final List<List<double>> _anchors;
  late final TensorFloat32Views _floatViews;

  ObjectDetection._(
    this._itp,
    this._inW,
    this._inH,
  );

  /// The model input width in pixels.
  int get inputWidth => _inW;

  /// The model input height in pixels.
  int get inputHeight => _inH;

  /// Total number of anchor boxes.
  int get numAnchors => _binding.numAnchors;

  /// Number of classes the model can detect (excluding placeholders).
  int get numClasses => _binding.numClasses;

  /// Creates and initializes an object detection model from package assets.
  ///
  /// The [model] parameter selects which TFLite model variant to load.
  /// The [performanceConfig] enables hardware acceleration delegates.
  static Future<ObjectDetection> create(
    ObjectDetectionModel model, {
    InterpreterOptions? options,
    PerformanceConfig? performanceConfig,
  }) =>
      _createWithLoader(
        model: model,
        load: (opts) => Interpreter.fromAsset(
          'packages/object_detection/assets/models/${_nameFor(model)}',
          options: opts,
        ),
        options: options,
        performanceConfig: performanceConfig,
      );

  /// Creates an object detection model from pre-loaded TFLite bytes.
  ///
  /// Used internally by [ObjectDetector] when initializing models inside a
  /// background isolate.
  static Future<ObjectDetection> createFromBuffer(
    Uint8List modelBytes,
    ObjectDetectionModel model, {
    PerformanceConfig? performanceConfig,
  }) =>
      _createWithLoader(
        model: model,
        load: (opts) => Interpreter.fromBuffer(modelBytes, options: opts),
        performanceConfig: performanceConfig,
      );

  static Future<ObjectDetection> _createWithLoader({
    required ObjectDetectionModel model,
    required FutureOr<Interpreter> Function(InterpreterOptions) load,
    InterpreterOptions? options,
    PerformanceConfig? performanceConfig,
  }) async {
    Delegate? delegate;
    final InterpreterOptions interpreterOptions;
    if (options != null) {
      interpreterOptions = options;
    } else {
      final result = InterpreterFactory.create(performanceConfig);
      interpreterOptions = result.$1;
      delegate = result.$2;
    }

    final Interpreter itp = await load(interpreterOptions);
    final List<int> ishape = itp.getInputTensor(0).shape;
    final int inH = ishape[1];
    final int inW = ishape[2];
    itp.allocateTensors();

    final ObjectDetection obj = ObjectDetection._(itp, inW, inH);
    obj._delegate = delegate;
    obj._binding = obj._discoverOutputBinding();

    // Generate anchors once at load time.
    obj._anchors = generateEfficientDetAnchors(imageSize: inW);
    if (obj._anchors.length != obj._binding.numAnchors) {
      throw StateError(
        'Anchor count mismatch: generator produced ${obj._anchors.length} '
        'anchors, model expects ${obj._binding.numAnchors}. '
        'Input size: ${inW}x$inH.',
      );
    }

    obj._floatViews = TensorFloat32Views.capture(itp);

    return obj;
  }

  /// Inspects the loaded interpreter's output tensors to identify which
  /// output index is boxes vs class scores by shape:
  ///   - 3D last-dim 4  → boxes
  ///   - 3D last-dim >4 → class scores
  _DetectorOutputBinding _discoverOutputBinding() {
    final List<Tensor> outs = _itp.getOutputTensors();
    int? boxesIdx;
    int? classesIdx;
    int? numAnchors;
    int? numClasses;

    for (int i = 0; i < outs.length; i++) {
      final shape = outs[i].shape;
      if (shape.length == 3) {
        if (shape[2] == 4) {
          boxesIdx = i;
          numAnchors = shape[1];
        } else if (shape[2] > 4) {
          classesIdx = i;
          numClasses = shape[2];
          numAnchors ??= shape[1];
        }
      }
    }

    if (boxesIdx == null ||
        classesIdx == null ||
        numAnchors == null ||
        numClasses == null) {
      throw StateError(
        'Could not identify object-detector output tensors '
        '(boxes=$boxesIdx classes=$classesIdx). Got '
        '${outs.map((t) => t.shape).toList()}.',
      );
    }

    return _DetectorOutputBinding(
      boxesIdx: boxesIdx,
      classesIdx: classesIdx,
      numAnchors: numAnchors,
      numClasses: numClasses,
    );
  }

  /// Runs detection on a pre-letterboxed float32 tensor.
  ///
  /// Returns raw normalized detections (in the model-input coordinate space,
  /// post letterbox-removal). Filtering by score / category / max-results
  /// happens upstream in the core.
  Future<List<Detection>> callWithTensor(
    ImageTensor pack, {
    double scoreThreshold = 0.0,
  }) async {
    _floatViews.inputs[0].setAll(0, pack.tensorNHWC);
    _itp.invoke();

    final boxesT = _itp.getOutputTensor(_binding.boxesIdx);
    final classesT = _itp.getOutputTensor(_binding.classesIdx);
    final Float32List boxBuf = boxesT.data.buffer.asFloat32List();
    final Float32List clsBuf = classesT.data.buffer.asFloat32List();

    final dets = _decodeAnchorsAndScore(
      boxBuf: boxBuf,
      clsBuf: clsBuf,
      scoreThreshold: scoreThreshold,
    );

    // Run NMS at IoU 0.45 (MediaPipe default) and cap at 200 candidates.
    final boxes = dets
        .map((d) => [
              d.boundingBox.xmin,
              d.boundingBox.ymin,
              d.boundingBox.xmax,
              d.boundingBox.ymax,
            ])
        .toList();
    final scores = dets.map((d) => d.score).toList();
    final pruned = weightedNms(boxes, scores, iouThres: 0.45, maxDet: 200);

    final List<Detection> kept = [];
    for (final r in pruned) {
      final src = dets[r.index];
      kept.add(
        Detection(
          boundingBox: RectF(r.box[0], r.box[1], r.box[2], r.box[3]),
          score: r.score,
          classIndex: src.classIndex,
        ),
      );
    }

    return _detectionLetterboxRemoval(kept, pack.padding);
  }

  /// Iterates anchors, finds top class per anchor (sigmoid of logits), filters
  /// by [scoreThreshold], and decodes box deltas to normalized `[xmin, ymin,
  /// xmax, ymax]` coordinates in model-input space.
  List<Detection> _decodeAnchorsAndScore({
    required Float32List boxBuf,
    required Float32List clsBuf,
    required double scoreThreshold,
  }) {
    final int n = _binding.numAnchors;
    final int k = _binding.numClasses;
    final List<Detection> out = <Detection>[];

    // Pre-image-tensor lookup of sigmoid threshold to short-circuit the inner
    // loop: any logit below this can't pass the score threshold.
    // sigmoid(x) >= s ⇒ x >= -ln((1-s)/s) = ln(s/(1-s))
    final double minLogit = scoreThreshold > 0 && scoreThreshold < 1
        ? math.log(scoreThreshold / (1.0 - scoreThreshold))
        : -1e9;

    for (int i = 0; i < n; i++) {
      final int classBase = i * k;
      // Find top class.
      double bestLogit = -double.infinity;
      int bestCls = -1;
      for (int c = 0; c < k; c++) {
        final double v = clsBuf[classBase + c];
        if (v > bestLogit) {
          bestLogit = v;
          bestCls = c;
        }
      }
      if (bestLogit < minLogit) continue;
      final double score = sigmoid(bestLogit);
      if (score < scoreThreshold) continue;

      final List<double> a = _anchors[i];
      final double cxA = a[0], cyA = a[1], wA = a[2], hA = a[3];
      final int boxBase = i * 4;
      // EfficientDet outputs [ty, tx, th, tw] (y first then x, RetinaNet style).
      final double ty = boxBuf[boxBase + 0];
      final double tx = boxBuf[boxBase + 1];
      final double th = boxBuf[boxBase + 2];
      final double tw = boxBuf[boxBase + 3];
      final double cy = ty * hA + cyA;
      final double cx = tx * wA + cxA;
      final double h = math.exp(th) * hA;
      final double w = math.exp(tw) * wA;

      // Clip to [0, 1] so degenerate / off-screen boxes get caught here
      // rather than surviving NMS (NMS won't merge them away because clipped
      // off-screen boxes have no IoU overlap with valid boxes).
      double xmin = cx - w * 0.5;
      double ymin = cy - h * 0.5;
      double xmax = cx + w * 0.5;
      double ymax = cy + h * 0.5;
      if (xmin < 0.0) xmin = 0.0;
      if (ymin < 0.0) ymin = 0.0;
      if (xmax > 1.0) xmax = 1.0;
      if (ymax > 1.0) ymax = 1.0;
      const double minEdge = 1e-3;
      if (xmax - xmin < minEdge || ymax - ymin < minEdge) continue;
      out.add(
        Detection(
          boundingBox: RectF(xmin, ymin, xmax, ymax),
          score: score,
          classIndex: bestCls,
        ),
      );
    }

    return out;
  }

  /// Releases TensorFlow Lite resources.
  void dispose() => _doDispose();
}
