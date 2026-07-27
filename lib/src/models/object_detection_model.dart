part of '../../object_detection.dart';

/// Output binding for an EfficientDet detector.
///
/// Both EfficientDet TFLite outputs are 3D `[1, numAnchors, X]`:
///   - boxes: shape `[1, A, 4]`, `[ty, tx, th, tw]` deltas relative to anchor.
///   - classes: shape `[1, A, K]`, per-class probabilities (the model output
///     tensor is produced by a LOGISTIC op).
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
/// These TFLite files emit RetinaNet-style anchor outputs with activated class
/// probabilities; this class generates the anchors at load time, then applies
/// per-anchor argmax + box decoding + weighted NMS in Dart to produce final
/// detections.
///
/// Most users should use the high-level [ObjectDetector] class instead of
/// working with this low-level model API directly.
class ObjectDetection with _TfliteModelDisposable {
  @override
  final Interpreter? _itp;

  /// LiteRT Next engine, used instead of [_itp] when the model was created
  /// with [createCompiledFromBuffer]. Exactly one of the two is non-null.
  @override
  final CompiledModel? _compiled;

  final int _inW, _inH;
  late final _DetectorOutputBinding _binding;

  /// Anchors packed as `[cx, cy, w, h]` per anchor. Flat so the decode loop
  /// walks one contiguous buffer instead of dereferencing one `List<double>`
  /// per anchor (19 206 of them for Lite0, 37 629 for Lite2).
  late final Float32List _anchors;
  late final TensorFloat32Views _floatViews;

  /// Scratch buffers reused across [callWithTensor] calls so a steady-state
  /// detection loop allocates nothing per frame. Grown on demand; the decode
  /// step writes survivors densely into the leading slots.
  Float32List _decodedBoxes = Float32List(0);
  Float32List _decodedScores = Float32List(0);
  Int32List _decodedClasses = Int32List(0);

  ObjectDetection._interpreter(
    Interpreter itp,
    this._inW,
    this._inH,
  )   : _itp = itp,
        _compiled = null;

  ObjectDetection._compiled(
    CompiledModel compiled,
    this._inW,
    this._inH,
  )   : _itp = null,
        _compiled = compiled;

  Interpreter _requireInterpreter() {
    final itp = _itp;
    if (itp == null) {
      throw StateError(
        'This ObjectDetection is backed by the CompiledModel engine and has '
        'no Interpreter.',
      );
    }
    return itp;
  }

  /// Whether inference runs on the LiteRT Next [CompiledModel] engine rather
  /// than the classic [Interpreter] + delegate path.
  bool get usesCompiledModel => _compiled != null;

  /// The accelerators the [CompiledModel] engine actually compiled to, or null
  /// on the interpreter path.
  ///
  /// Read this after initialization to find out whether a requested GPU
  /// compilation succeeded or silently fell back to CPU.
  Set<Accelerator>? get activeAccelerators => _compiled?.accelerators;

  /// The model input width in pixels.
  int get inputWidth => _inW;

  /// The model input height in pixels.
  int get inputHeight => _inH;

  /// A reusable input tensor buffer sized for this model.
  ///
  /// Pass it as `convertImageToTensor(..., buffer: model.newInputBuffer())` and
  /// keep it alive across frames to avoid reallocating `inputWidth *
  /// inputHeight * 3` floats (1.2 MB for Lite0, 2.4 MB for Lite2) per call.
  Float32List newInputBuffer() => Float32List(_inW * _inH * 3);

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

  /// Creates an object detection model backed by the LiteRT Next
  /// [CompiledModel] engine instead of an [Interpreter] plus delegate.
  ///
  /// [accelerators] defaults to "try GPU, fall back to CPU". Any other set is
  /// treated as a hard requirement and will throw rather than degrade.
  /// [precision] selects the GPU compute precision and is ignored on CPU.
  ///
  /// Read [activeAccelerators] afterwards to see what the model actually
  /// compiled to.
  static Future<ObjectDetection> createCompiledFromBuffer(
    Uint8List modelBytes,
    ObjectDetectionModel model, {
    Set<Accelerator> accelerators = const {Accelerator.gpu, Accelerator.cpu},
    Precision precision = Precision.fp16,
    bool forceCpu = false,
    TensorBufferMode tensorBufferMode = TensorBufferMode.hostMemory,
  }) async {
    // Host-memory buffers let the decoder read the detection heads as views of
    // model-owned memory. That matters far more here than for a landmark model:
    // EfficientDet emits ~1.8M floats per frame (Lite0) or ~3.4M (Lite2), so
    // the managed path's copy-out is 7-14 MB of fresh Dart allocation per
    // frame. Measured on macOS/Metal, zero-copy is ~1.4x the managed path.
    //
    // The mode is not universally supported, so a failure falls back to
    // managed buffers rather than failing initialization.
    CompiledModel? compiled;
    if (tensorBufferMode == TensorBufferMode.hostMemory) {
      try {
        compiled = compiledModelFromBufferAuto(
          modelBytes,
          accelerators: accelerators,
          precision: precision,
          forceCpu: forceCpu,
          tensorBufferMode: TensorBufferMode.hostMemory,
          onGpuFallback: _onGpuFallback,
        );
      } catch (e) {
        debugPrint(
          'object_detection: host-memory tensor buffers unavailable, using '
          'managed buffers. Error: $e',
        );
      }
    }
    compiled ??= compiledModelFromBufferAuto(
      modelBytes,
      accelerators: accelerators,
      precision: precision,
      forceCpu: forceCpu,
      onGpuFallback: _onGpuFallback,
    );

    ObjectDetection? obj;
    try {
      // CompiledModel reports tensor sizes in bytes only (there is no
      // Interpreter to query shapes from), so the geometry is re-derived from
      // the byte sizes: the input is a square [1, S, S, 3] tensor, and the two
      // outputs are distinguished by how many floats each carries per anchor.
      final int side = compiledSquareInputSide(
        compiled,
        label: 'object detection',
      );
      obj = ObjectDetection._compiled(compiled, side, side);
      obj._anchors = generateEfficientDetAnchorsFlat(imageSize: side);
      final int anchorCount = obj._anchors.length ~/ 4;

      final List<int> counts = compiledOutputFloatCounts(
        compiled,
        label: 'object detection',
      );
      final int boxesIdx =
          indexWhereFloatCount(counts, (f) => f == anchorCount * 4);
      final int classesIdx = indexWhereFloatCount(
        counts,
        (f) => f % anchorCount == 0 && f ~/ anchorCount > 4,
      );
      if (boxesIdx < 0 || classesIdx < 0) {
        throw UnsupportedError(
          'Could not identify compiled object-detector outputs for '
          '$anchorCount anchors. Output float counts: $counts.',
        );
      }

      obj._binding = _DetectorOutputBinding(
        boxesIdx: boxesIdx,
        classesIdx: classesIdx,
        numAnchors: anchorCount,
        numClasses: counts[classesIdx] ~/ anchorCount,
      );
      return obj;
    } catch (_) {
      if (obj != null) {
        obj.dispose();
      } else {
        compiled.close();
      }
      rethrow;
    }
  }

  static void _onGpuFallback(Object error) {
    debugPrint(
      'object_detection: GPU CompiledModel compilation failed, falling back '
      'to CPU. Error: $error',
    );
  }

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

    final ObjectDetection obj = ObjectDetection._interpreter(itp, inW, inH);
    obj._delegate = delegate;
    obj._binding = obj._discoverOutputBinding();

    // Generate anchors once at load time.
    obj._anchors = generateEfficientDetAnchorsFlat(imageSize: inW);
    if (obj._anchors.length ~/ 4 != obj._binding.numAnchors) {
      throw StateError(
        'Anchor count mismatch: generator produced '
        '${obj._anchors.length ~/ 4} anchors, model expects '
        '${obj._binding.numAnchors}. Input size: ${inW}x$inH.',
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
    final List<Tensor> outs = _requireInterpreter().getOutputTensors();
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
    final int candidates;

    final CompiledModel? compiled = _compiled;
    if (compiled == null) {
      final Interpreter itp = _requireInterpreter();
      _floatViews.inputs[0].setAll(0, pack.tensorNHWC);
      itp.invoke();
      candidates = _decodeAnchorsAndScore(
        boxBuf:
            itp.getOutputTensor(_binding.boxesIdx).data.buffer.asFloat32List(),
        clsBuf: itp
            .getOutputTensor(_binding.classesIdx)
            .data
            .buffer
            .asFloat32List(),
        scoreThreshold: scoreThreshold,
      );
    } else if (compiled.tensorBufferMode == TensorBufferMode.hostMemory) {
      // Zero-copy: write into the model's aligned host memory, run, and decode
      // straight out of the output buffers. Nothing in this path materializes
      // the multi-megabyte detection heads as Dart lists.
      //
      // dispatch() blocks, matching the interpreter branch above (invoke() is
      // synchronous too). The detector runs this inside its own background
      // isolate, where blocking is the point and runAsync's helper isolate
      // would only add a hop.
      compiled.writeInput(0, (dst) => dst.setAll(0, pack.tensorNHWC));
      compiled.dispatch();
      candidates = compiled.readOutput(
        _binding.boxesIdx,
        (boxBuf) => compiled.readOutput(
          _binding.classesIdx,
          (clsBuf) => _decodeAnchorsAndScore(
            boxBuf: boxBuf,
            clsBuf: clsBuf,
            scoreThreshold: scoreThreshold,
          ),
        ),
      );
    } else {
      final List<Float32List> outputs = compiled.run([pack.tensorNHWC]);
      candidates = _decodeAnchorsAndScore(
        boxBuf: outputs[_binding.boxesIdx],
        clsBuf: outputs[_binding.classesIdx],
        scoreThreshold: scoreThreshold,
      );
    }

    // Run NMS at IoU 0.45 (MediaPipe default) and cap at 200 candidates.
    final boxes = List<List<double>>.generate(
      candidates,
      (i) => <double>[
        _decodedBoxes[i * 4],
        _decodedBoxes[i * 4 + 1],
        _decodedBoxes[i * 4 + 2],
        _decodedBoxes[i * 4 + 3],
      ],
      growable: false,
    );
    final scores = List<double>.generate(
      candidates,
      (i) => _decodedScores[i],
      growable: false,
    );
    final pruned = weightedNms(boxes, scores, iouThres: 0.45, maxDet: 200);

    final List<Detection> kept = [];
    for (final r in pruned) {
      kept.add(
        Detection(
          boundingBox: RectF(r.box[0], r.box[1], r.box[2], r.box[3]),
          score: r.score,
          classIndex: _decodedClasses[r.index],
        ),
      );
    }

    return _detectionLetterboxRemoval(kept, pack.padding);
  }

  /// Ensures the decode scratch buffers can hold [capacity] candidates,
  /// preserving the [used] entries already written.
  void _ensureDecodeCapacity(int capacity, int used) {
    if (_decodedScores.length >= capacity) return;
    // Grow geometrically so a scene that gets busier does not reallocate on
    // every frame.
    int next = _decodedScores.isEmpty ? 256 : _decodedScores.length;
    while (next < capacity) {
      next *= 2;
    }
    _decodedBoxes = Float32List(next * 4)..setRange(0, used * 4, _decodedBoxes);
    _decodedScores = Float32List(next)..setRange(0, used, _decodedScores);
    _decodedClasses = Int32List(next)..setRange(0, used, _decodedClasses);
  }

  /// Iterates anchors, finds the top class probability per anchor, filters by
  /// [scoreThreshold], and decodes box deltas to normalized `[xmin, ymin,
  /// xmax, ymax]` coordinates in model-input space.
  ///
  /// Survivors are written densely into [_decodedBoxes] / [_decodedScores] /
  /// [_decodedClasses]; the return value is how many were written.
  ///
  /// The argmax is seeded at [scoreThreshold] rather than negative infinity:
  /// the overwhelming majority of anchors have no class anywhere near the bar,
  /// so their entire inner loop is a load plus a not-taken compare and never
  /// touches the running best. Anchors that do clear the bar fall into the
  /// slower branch, which resolves ties to the lowest class index exactly as
  /// an unseeded `max` scan would.
  int _decodeAnchorsAndScore({
    required Float32List boxBuf,
    required Float32List clsBuf,
    required double scoreThreshold,
  }) {
    final int n = _binding.numAnchors;
    final int k = _binding.numClasses;
    final Float32List anchors = _anchors;
    int count = 0;

    for (int i = 0; i < n; i++) {
      // Find the top class, but only among those at or above the threshold.
      final int classBase = i * k;
      final int classEnd = classBase + k;
      double bestScore = scoreThreshold;
      int bestCls = -1;
      for (int p = classBase; p < classEnd; p++) {
        if (clsBuf[p] >= bestScore) {
          final double v = clsBuf[p];
          if (bestCls < 0 || v > bestScore) {
            bestScore = v;
            bestCls = p - classBase;
          }
        }
      }
      if (bestCls < 0) continue;

      final int aBase = i * 4;
      final double cxA = anchors[aBase];
      final double cyA = anchors[aBase + 1];
      final double wA = anchors[aBase + 2];
      final double hA = anchors[aBase + 3];
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

      _ensureDecodeCapacity(count + 1, count);
      final int o = count * 4;
      _decodedBoxes[o] = xmin;
      _decodedBoxes[o + 1] = ymin;
      _decodedBoxes[o + 2] = xmax;
      _decodedBoxes[o + 3] = ymax;
      _decodedScores[count] = bestScore;
      _decodedClasses[count] = bestCls;
      count++;
    }

    return count;
  }

  /// Releases TensorFlow Lite resources.
  void dispose() => _doDispose();
}
