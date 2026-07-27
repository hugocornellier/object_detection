part of '../../object_detection.dart';

/// Holds metadata for an output tensor (shape plus its writable buffer).
@Deprecated(
  'Unused by this package. Use collectOutputShapes from flutter_litert, which '
  'returns shapes without materializing tensor buffers. Will be removed in the '
  'next major release.',
)
class OutputTensorInfo {
  /// Creates an [OutputTensorInfo] with the given [shape] and [buffer].
  ///
  /// The [shape] describes the tensor dimensions and [buffer] provides
  /// direct access to the tensor's underlying Float32 data.
  OutputTensorInfo(this.shape, this.buffer);

  /// The dimensions of the tensor.
  final List<int> shape;

  /// The underlying Float32 buffer containing the tensor's raw data.
  final Float32List buffer;
}

/// Collects output tensor shapes (and their backing buffers) for an interpreter.
@Deprecated(
  'Unused by this package. Use collectOutputShapes from flutter_litert. Will '
  'be removed in the next major release.',
)
Map<int, OutputTensorInfo> collectOutputTensorInfo(Interpreter itp) {
  final Map<int, OutputTensorInfo> outputs = <int, OutputTensorInfo>{};
  for (int i = 0;; i++) {
    try {
      final Tensor t = itp.getOutputTensor(i);
      outputs[i] = OutputTensorInfo(t.shape, t.data.buffer.asFloat32List());
    } catch (_) {
      break;
    }
  }
  return outputs;
}

/// Test-only access to [collectOutputTensorInfo] for verifying output tensor collection.
@visibleForTesting
@Deprecated(
  'Accompanies collectOutputTensorInfo. Will be removed in the next major '
  'release.',
)
Map<int, OutputTensorInfo> testCollectOutputTensorInfo(Interpreter itp) =>
    // ignore: deprecated_member_use_from_same_package
    collectOutputTensorInfo(itp);

/// Shared dispose logic for TFLite model classes.
///
/// [_itp] is null when the model is backed by the LiteRT Next [CompiledModel]
/// engine instead of an [Interpreter]; [_compiled] is null on the interpreter
/// path. Exactly one of the two is set.
mixin _TfliteModelDisposable {
  IsolateInterpreter? _iso;
  Delegate? _delegate;
  bool _disposed = false;

  Interpreter? get _itp;
  CompiledModel? get _compiled;

  void _doDispose() {
    if (_disposed) return;
    _disposed = true;
    _delegate?.delete();
    _delegate = null;
    _iso?.close();
    _itp?.close();
    _compiled?.close();
  }
}

String _nameFor(ObjectDetectionModel m) {
  switch (m) {
    case ObjectDetectionModel.efficientDetLite0:
      return _modelNameLite0;
    case ObjectDetectionModel.efficientDetLite2:
      return _modelNameLite2;
  }
}

/// Generates EfficientDet RetinaNet-style multi-scale anchors, packed into a
/// flat `[cx, cy, w, h]` [Float32List] of length `anchorCount * 4`.
///
/// This is the layout the decoder actually consumes. Lite0 has 19 206 anchors
/// and Lite2 has 37 629, so the nested `List<List<double>>` shape costs one
/// heap object plus one bounds-checked indirection per anchor; the flat buffer
/// is a single allocation the hot loop can walk sequentially.
///
/// See [generateEfficientDetAnchors] for the nested-list view of the same data.
Float32List generateEfficientDetAnchorsFlat({
  required int imageSize,
  int minLevel = 3,
  int maxLevel = 7,
  int numScales = 3,
  List<double> aspectRatios = const [1.0, 2.0, 0.5],
  double anchorScale = 4.0,
}) {
  int count = 0;
  for (int level = minLevel; level <= maxLevel; level++) {
    final int featureSize = (imageSize / (1 << level)).ceil();
    count += featureSize * featureSize * numScales * aspectRatios.length;
  }

  final Float32List anchors = Float32List(count * 4);
  int w = 0;
  for (int level = minLevel; level <= maxLevel; level++) {
    final int stride = 1 << level;
    final int featureSize = (imageSize / stride).ceil();
    final double baseAnchorSize = anchorScale * stride.toDouble();
    for (int y = 0; y < featureSize; y++) {
      final double cy = (y + 0.5) * stride / imageSize;
      for (int x = 0; x < featureSize; x++) {
        final double cx = (x + 0.5) * stride / imageSize;
        for (int s = 0; s < numScales; s++) {
          final double scale = math.pow(2, s / numScales).toDouble();
          for (final aspect in aspectRatios) {
            final double sqAspect = math.sqrt(aspect);
            anchors[w++] = cx;
            anchors[w++] = cy;
            anchors[w++] = baseAnchorSize * scale * sqAspect / imageSize;
            anchors[w++] = baseAnchorSize * scale / sqAspect / imageSize;
          }
        }
      }
    }
  }
  return anchors;
}

/// Generates EfficientDet RetinaNet-style multi-scale anchors.
///
/// EfficientDet uses 5 feature pyramid levels (P3-P7) with `numScales` (3) ×
/// `aspectRatios.length` (3) = 9 anchors per spatial location. Anchors are
/// returned in normalized image coordinates as `[cx, cy, w, h]`.
///
/// For Lite0 with `imageSize=320`, total anchors = 19 206.
/// For Lite2 with `imageSize=448`, total anchors = 37 629.
///
/// The detector itself uses [generateEfficientDetAnchorsFlat]; this nested-list
/// view is kept for callers that want to inspect anchors one at a time.
List<List<double>> generateEfficientDetAnchors({
  required int imageSize,
  int minLevel = 3,
  int maxLevel = 7,
  int numScales = 3,
  List<double> aspectRatios = const [1.0, 2.0, 0.5],
  double anchorScale = 4.0,
}) {
  final Float32List flat = generateEfficientDetAnchorsFlat(
    imageSize: imageSize,
    minLevel: minLevel,
    maxLevel: maxLevel,
    numScales: numScales,
    aspectRatios: aspectRatios,
    anchorScale: anchorScale,
  );
  return List<List<double>>.generate(
    flat.length ~/ 4,
    (i) => <double>[flat[i * 4], flat[i * 4 + 1], flat[i * 4 + 2],
        flat[i * 4 + 3]],
    growable: false,
  );
}

/// Test-only access to anchor generation.
@visibleForTesting
List<List<double>> testGenerateEfficientDetAnchors({
  required int imageSize,
}) =>
    generateEfficientDetAnchors(imageSize: imageSize);

/// Test-only: exposes the private model-name mapping for unit tests.
@visibleForTesting
String testNameFor(ObjectDetectionModel m) => _nameFor(m);

/// Reads the bundled COCO labelmap (`labelmap.txt`) from package assets.
///
/// Returns the list of label strings, indexable by class index. Some entries
/// are placeholder `???` strings to keep alignment with the original COCO IDs.
Future<List<String>> loadLabelMap() async {
  final raw = await rootBundle.loadString(
    'packages/object_detection/assets/models/$_labelMapAsset',
  );
  return raw
      .split('\n')
      .map((s) => s.trim())
      .where((s) => s.isNotEmpty)
      .toList(growable: false);
}

/// Parses labelmap text content into a list of strings. Useful for tests and
/// when label data has already been loaded from another source.
@visibleForTesting
List<String> parseLabelMap(String content) => content
    .split('\n')
    .map((s) => s.trim())
    .where((s) => s.isNotEmpty)
    .toList(growable: false);

/// Converts a continuous BGR `CV_8UC3` [cv.Mat] to a `[-1, 1]`-normalized RGB
/// float tensor using OpenCV's SIMD kernels (BGR→RGB swap plus scaled float
/// conversion) followed by a single bulk copy into [buffer].
///
/// Falls back to the scalar Dart loop ([bgrBytesToSignedFloat32]) for
/// non-`CV_8UC3` or non-continuous inputs, which produces identical values.
@visibleForTesting
Float32List bgrMatToSignedFloat32(
  cv.Mat mat, {
  required int totalPixels,
  Float32List? buffer,
}) {
  // BGRA (4ch) and grayscale (1ch) inputs must be colour-converted to 3-channel
  // BGR first: both the SIMD path below and the byte fallback assume 3
  // bytes/pixel, so a non-BGR stride overruns the buffer (BGRA) or under-reads
  // it (grayscale) and corrupts the tensor.
  cv.Mat? owned;
  cv.Mat src = mat;
  if (mat.type == cv.MatType.CV_8UC4) {
    src = owned = cv.cvtColor(mat, cv.COLOR_BGRA2BGR);
  } else if (mat.type == cv.MatType.CV_8UC1) {
    src = owned = cv.cvtColor(mat, cv.COLOR_GRAY2BGR);
  }
  try {
    if (src.type != cv.MatType.CV_8UC3 || !src.isContinuous) {
      return bgrBytesToSignedFloat32(
        bytes: src.data,
        totalPixels: totalPixels,
        buffer: buffer,
      );
    }
    final cv.Mat rgb = cv.cvtColor(src, cv.COLOR_BGR2RGB);
    final cv.Mat f32 = rgb.convertTo(
      cv.MatType.CV_32FC3,
      alpha: 1.0 / 127.5,
      beta: -1.0,
    );
    rgb.dispose();
    final Uint8List raw = f32.data;
    final Float32List view = Float32List.view(
      raw.buffer,
      raw.offsetInBytes,
      totalPixels * 3,
    );
    final Float32List tensor = buffer ?? Float32List(totalPixels * 3);
    tensor.setAll(0, view);
    f32.dispose();
    return tensor;
  } finally {
    owned?.dispose();
  }
}

/// Converts a cv.Mat image to a normalized float32 tensor with letterboxing.
///
/// Performs aspect-preserving resize with black padding and normalizes pixel
/// values to the `[-1.0, 1.0]` range expected by EfficientDet float32/float16
/// models (mean=127.5, std=127.5). Channel order is BGR→RGB.
///
/// Pass [buffer] to write into a caller-owned tensor and avoid allocating a
/// fresh `outW * outH * 3` [Float32List] on every frame.
///
/// The input cv.Mat is NOT disposed by this function.
ImageTensor convertImageToTensor(
  cv.Mat src, {
  required int outW,
  required int outH,
  Float32List? buffer,
}) {
  final int inW = src.cols;
  final int inH = src.rows;

  final LetterboxParams lbp = computeLetterboxParams(
    srcWidth: inW,
    srcHeight: inH,
    targetWidth: outW,
    targetHeight: outH,
  );

  // Skip the resize when the source already matches the target geometry, and
  // skip the border copy when the letterbox is degenerate (square source into
  // a square model input). A non-continuous source still goes through
  // cv.resize so the conversion below always reads tightly packed rows.
  final bool needsResize =
      inW != lbp.newWidth || inH != lbp.newHeight || !src.isContinuous;
  final cv.Mat resized = needsResize
      ? cv.resize(
          src,
          (lbp.newWidth, lbp.newHeight),
          interpolation: cv.INTER_LINEAR,
        )
      : src;

  final bool needsPad = lbp.padTop != 0 ||
      lbp.padBottom != 0 ||
      lbp.padLeft != 0 ||
      lbp.padRight != 0;
  final cv.Mat padded = needsPad
      ? cv.copyMakeBorder(
          resized,
          lbp.padTop,
          lbp.padBottom,
          lbp.padLeft,
          lbp.padRight,
          cv.BORDER_CONSTANT,
          value: cv.Scalar.black,
        )
      : resized;
  if (needsResize && needsPad) resized.dispose();

  final Float32List tensor = bgrMatToSignedFloat32(
    padded,
    totalPixels: outW * outH,
    buffer: buffer,
  );
  if (needsResize || needsPad) padded.dispose();

  final double padTopNorm = lbp.padTop / outH;
  final double padBottomNorm = lbp.padBottom / outH;
  final double padLeftNorm = lbp.padLeft / outW;
  final double padRightNorm = lbp.padRight / outW;

  return ImageTensor(
    tensor,
    [padTopNorm, padBottomNorm, padLeftNorm, padRightNorm],
    outW,
    outH,
  );
}

/// Removes letterbox padding from normalized detection coordinates.
///
/// Detections come out of the model in coordinates `[0, 1]` relative to the
/// letterboxed model input. This rescales them back to the source image's
/// `[0, 1]` coordinate space by dividing by the unpadded fraction.
List<Detection> _detectionLetterboxRemoval(
  List<Detection> dets,
  List<double> padding,
) {
  final double pt = padding[0],
      pb = padding[1],
      pl = padding[2],
      pr = padding[3];
  final double sx = 1.0 - (pl + pr);
  final double sy = 1.0 - (pt + pb);
  if (sx <= 0 || sy <= 0) return dets;

  double clamp01(double v) => v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);

  RectF unpad(RectF r) => RectF(
        clamp01((r.xmin - pl) / sx),
        clamp01((r.ymin - pt) / sy),
        clamp01((r.xmax - pl) / sx),
        clamp01((r.ymax - pt) / sy),
      );

  // Boxes that landed in the padding region collapse to zero width or
  // height after clamping; drop them so callers never see degenerate boxes.
  const double minEdge = 1e-4;

  final result = <Detection>[];
  for (final d in dets) {
    final unpadded = unpad(d.boundingBox);
    if (unpadded.xmax - unpadded.xmin < minEdge ||
        unpadded.ymax - unpadded.ymin < minEdge) {
      continue;
    }
    result.add(
      Detection(
        boundingBox: unpadded,
        score: d.score,
        classIndex: d.classIndex,
        imageSize: d.imageSize,
      ),
    );
  }
  return result;
}

/// Test-only: exposes the private letterbox-removal logic for unit tests.
@visibleForTesting
List<Detection> testDetectionLetterboxRemoval(
  List<Detection> dets,
  List<double> padding,
) =>
    _detectionLetterboxRemoval(dets, padding);

/// Applies score threshold, category allow/deny lists, and max-results cap.
///
/// Filters [detections] in place per the per-call [options], then sorts by
/// descending score and trims to `options.maxResults` if set.
List<Detection> _applyOptions(
  List<Detection> detections,
  ObjectDetectorOptions options,
  List<String> labels,
) {
  if (options.categoryAllowlist.isNotEmpty &&
      options.categoryDenylist.isNotEmpty) {
    throw ArgumentError(
      'categoryAllowlist and categoryDenylist are mutually exclusive. '
      'Pass at most one.',
    );
  }

  final allow = options.categoryAllowlist;
  final deny = options.categoryDenylist;
  final filtered = <Detection>[];
  for (final d in detections) {
    if (d.score < options.scoreThreshold) continue;
    final name = d.classIndex >= 0 && d.classIndex < labels.length
        ? labels[d.classIndex]
        : '???';
    if (allow.isNotEmpty && !allow.contains(name)) continue;
    if (deny.isNotEmpty && deny.contains(name)) continue;
    filtered.add(d);
  }
  filtered.sort((a, b) => b.score.compareTo(a.score));
  final cap = options.maxResults;
  if (cap != null && cap >= 0 && filtered.length > cap) {
    return filtered.sublist(0, cap);
  }
  return filtered;
}

/// Test-only: exposes the private filter pipeline for unit tests.
@visibleForTesting
List<Detection> testApplyOptions(
  List<Detection> detections,
  ObjectDetectorOptions options,
  List<String> labels,
) =>
    _applyOptions(detections, options, labels);
