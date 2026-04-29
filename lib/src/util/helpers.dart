part of '../../object_detection.dart';

/// Holds metadata for an output tensor (shape plus its writable buffer).
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
Map<int, OutputTensorInfo> testCollectOutputTensorInfo(Interpreter itp) =>
    collectOutputTensorInfo(itp);

/// Shared dispose logic for TFLite model classes.
mixin _TfliteModelDisposable {
  IsolateInterpreter? _iso;
  Delegate? _delegate;
  bool _disposed = false;

  Interpreter get _itp;

  void _doDispose() {
    if (_disposed) return;
    _disposed = true;
    _delegate?.delete();
    _delegate = null;
    _iso?.close();
    _itp.close();
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

/// Generates EfficientDet RetinaNet-style multi-scale anchors.
///
/// EfficientDet uses 5 feature pyramid levels (P3-P7) with `numScales` (3) ×
/// `aspectRatios.length` (3) = 9 anchors per spatial location. Anchors are
/// returned in normalized image coordinates as `[cx, cy, w, h]`.
///
/// For Lite0 with `imageSize=320`, total anchors = 19 206.
/// For Lite2 with `imageSize=448`, total anchors = 37 629.
List<List<double>> generateEfficientDetAnchors({
  required int imageSize,
  int minLevel = 3,
  int maxLevel = 7,
  int numScales = 3,
  List<double> aspectRatios = const [1.0, 2.0, 0.5],
  double anchorScale = 4.0,
}) {
  final anchors = <List<double>>[];
  for (int level = minLevel; level <= maxLevel; level++) {
    final int stride = 1 << level;
    final int featureSize = (imageSize / stride).ceil();
    final double baseAnchorSize = anchorScale * stride.toDouble();
    for (int y = 0; y < featureSize; y++) {
      for (int x = 0; x < featureSize; x++) {
        final double cy = (y + 0.5) * stride / imageSize;
        final double cx = (x + 0.5) * stride / imageSize;
        for (int s = 0; s < numScales; s++) {
          final double scale = math.pow(2, s / numScales).toDouble();
          for (final aspect in aspectRatios) {
            final double sqAspect = math.sqrt(aspect);
            final double w = baseAnchorSize * scale * sqAspect / imageSize;
            final double h = baseAnchorSize * scale / sqAspect / imageSize;
            anchors.add([cx, cy, w, h]);
          }
        }
      }
    }
  }
  return anchors;
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

/// Converts a cv.Mat image to a normalized float32 tensor with letterboxing.
///
/// Performs aspect-preserving resize with black padding and normalizes pixel
/// values to the `[-1.0, 1.0]` range expected by EfficientDet float32/float16
/// models (mean=127.5, std=127.5). Channel order is BGR→RGB.
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

  final cv.Mat resized = cv.resize(
    src,
    (lbp.newWidth, lbp.newHeight),
    interpolation: cv.INTER_LINEAR,
  );

  final cv.Mat padded = cv.copyMakeBorder(
    resized,
    lbp.padTop,
    lbp.padBottom,
    lbp.padLeft,
    lbp.padRight,
    cv.BORDER_CONSTANT,
    value: cv.Scalar.black,
  );
  resized.dispose();

  final Float32List tensor = bgrBytesToSignedFloat32(
    bytes: padded.data,
    totalPixels: outW * outH,
    buffer: buffer,
  );
  padded.dispose();

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
