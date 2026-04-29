part of '../object_detection.dart';

/// Specifies which object detection model variant to use.
///
/// Different models trade off accuracy, speed, and size:
/// - [efficientDetLite0]: Default, 320×320 input. Good balance of speed and accuracy.
/// - [efficientDetLite2]: 448×448 input. Higher accuracy, slower.
///
/// Both models are 90-class COCO detectors (80 valid classes; the label map
/// has 10 placeholder slots to keep alignment with the original COCO IDs).
enum ObjectDetectionModel {
  efficientDetLite0,
  efficientDetLite2,
}

/// Per-call configuration for [ObjectDetector.detect] and friends.
///
/// Mirrors MediaPipe's `ObjectDetectorOptions`:
/// - [scoreThreshold]: Minimum confidence in `[0.0, 1.0]`. Defaults to 0.5.
/// - [maxResults]: Top-K cap after sorting by score. `null` (default) keeps all.
/// - [categoryAllowlist]: If non-empty, only detections whose category name is
///   in this list are kept. Mutually exclusive with [categoryDenylist].
/// - [categoryDenylist]: If non-empty, detections whose category name is in
///   this list are dropped. Mutually exclusive with [categoryAllowlist].
class ObjectDetectorOptions {
  /// Minimum confidence threshold (0.0 - 1.0). Detections below are dropped.
  final double scoreThreshold;

  /// Maximum number of detections to return (top-K by score). `null` for unlimited.
  final int? maxResults;

  /// If non-empty, only detections whose category name is in this list are kept.
  final List<String> categoryAllowlist;

  /// If non-empty, detections whose category name is in this list are dropped.
  final List<String> categoryDenylist;

  /// Creates a per-call options object. [categoryAllowlist] and [categoryDenylist]
  /// are mutually exclusive. Pass at most one.
  const ObjectDetectorOptions({
    this.scoreThreshold = 0.5,
    this.maxResults,
    this.categoryAllowlist = const <String>[],
    this.categoryDenylist = const <String>[],
  });

  /// Default options: 0.5 threshold, no cap, no category filtering.
  static const ObjectDetectorOptions defaults = ObjectDetectorOptions();

  /// Serializes this options object for isolate transfer.
  Map<String, dynamic> toMap() => {
        'scoreThreshold': scoreThreshold,
        'maxResults': maxResults,
        'categoryAllowlist': categoryAllowlist,
        'categoryDenylist': categoryDenylist,
      };

  /// Creates options from a serialized map.
  factory ObjectDetectorOptions.fromMap(Map<String, dynamic> map) =>
      ObjectDetectorOptions(
        scoreThreshold: (map['scoreThreshold'] as num).toDouble(),
        maxResults: map['maxResults'] as int?,
        categoryAllowlist:
            (map['categoryAllowlist'] as List?)?.cast<String>() ?? const [],
        categoryDenylist:
            (map['categoryDenylist'] as List?)?.cast<String>() ?? const [],
      );
}

/// A single category prediction for a detection.
///
/// Each [DetectedObject] carries one [Category] for object detection (the
/// model emits one class index + score per box). [index] indexes into the
/// label map; [categoryName] is the resolved label string; [score] is the
/// confidence.
///
/// `displayName` mirrors MediaPipe's locale-aware display string. We don't
/// ship locale label data, so `displayName` is always equal to [categoryName].
class Category {
  /// Class index in the label file.
  final int index;

  /// Confidence score (0.0 - 1.0).
  final double score;

  /// Resolved label string (e.g. "person", "car").
  final String categoryName;

  /// Locale-resolved display name. Equal to [categoryName] when no locale data is bundled.
  final String displayName;

  /// Creates a category with the given fields.
  const Category({
    required this.index,
    required this.score,
    required this.categoryName,
    required this.displayName,
  });

  /// Serializes this category for isolate transfer.
  Map<String, dynamic> toMap() => {
        'index': index,
        'score': score,
        'categoryName': categoryName,
        'displayName': displayName,
      };

  /// Creates a category from a serialized map.
  factory Category.fromMap(Map<String, dynamic> map) => Category(
        index: map['index'] as int,
        score: (map['score'] as num).toDouble(),
        categoryName: map['categoryName'] as String,
        displayName: map['displayName'] as String,
      );

  @override
  String toString() =>
      'Category($categoryName, score: ${score.toStringAsFixed(3)})';
}

/// A single detected object with bounding box and category.
///
/// Mirrors MediaPipe's `Detection` class for the Object Detector solution.
///
/// Example:
/// ```dart
/// final detections = await detector.detect(imageBytes);
/// for (final obj in detections) {
///   final box = obj.boundingBox;
///   final cat = obj.categories.first;
///   print('Found ${cat.categoryName} (${(cat.score * 100).toStringAsFixed(1)}%) '
///         'at (${box.topLeft.x.toInt()}, ${box.topLeft.y.toInt()})');
/// }
/// ```
class DetectedObject {
  final Detection _detection;

  /// All category predictions for this detection.
  ///
  /// For object detection from MediaPipe TFLite models the list always has
  /// exactly one entry (top-1 class), but the field is a list to mirror
  /// MediaPipe's API shape.
  final List<Category> categories;

  /// Dimensions of the original source image.
  ///
  /// Used internally to convert normalized bounding-box coordinates into
  /// pixel coordinates. Users typically don't need this directly.
  final Size originalSize;

  /// Creates a detected object with the given detection, categories, and image size.
  DetectedObject({
    required Detection detection,
    required this.categories,
    required this.originalSize,
  })  : _detection = detection,
        boundingBox = _computeBoundingBox(detection.boundingBox, originalSize);

  static BoundingBox _computeBoundingBox(RectF r, Size originalSize) {
    final double w = originalSize.width.toDouble();
    final double h = originalSize.height.toDouble();
    return BoundingBox(
      topLeft: Point(r.xmin * w, r.ymin * h),
      topRight: Point(r.xmax * w, r.ymin * h),
      bottomRight: Point(r.xmax * w, r.ymax * h),
      bottomLeft: Point(r.xmin * w, r.ymax * h),
    );
  }

  /// The detected object's bounding box in pixel coordinates.
  ///
  /// Provides convenient access to corner points, dimensions, and center.
  /// Use [BoundingBox.topLeft], [BoundingBox.topRight],
  /// [BoundingBox.bottomRight], [BoundingBox.bottomLeft] for individual corners,
  /// or [BoundingBox.width], [BoundingBox.height], [BoundingBox.center] for
  /// dimensions and center point.
  final BoundingBox boundingBox;

  /// The top-1 category for this detection (highest score).
  ///
  /// Equivalent to `categories.first` and always non-null for results from
  /// [ObjectDetector.detect].
  Category get category => categories.first;

  /// Convenience accessor for the top-1 category's score.
  double get score => category.score;

  /// Convenience accessor for the top-1 category's name.
  String get categoryName => category.categoryName;

  /// Serializes this detection for isolate transfer.
  Map<String, dynamic> toMap() => {
        'detection': _detection.toMap(),
        'categories': categories.map((c) => c.toMap()).toList(),
        'originalSize': {
          'width': originalSize.width,
          'height': originalSize.height,
        },
      };

  /// Creates a [DetectedObject] from a serialized map.
  factory DetectedObject.fromMap(Map<String, dynamic> map) => DetectedObject(
        detection: Detection.fromMap(map['detection']),
        categories: (map['categories'] as List)
            .map((c) => Category.fromMap(Map<String, dynamic>.from(c as Map)))
            .toList(),
        originalSize: Size(
          (map['originalSize']['width'] as num).toDouble(),
          (map['originalSize']['height'] as num).toDouble(),
        ),
      );

  @override
  String toString() =>
      'DetectedObject($categoryName, score: ${score.toStringAsFixed(3)}, '
      'box: ${boundingBox.topLeft.x.toInt()},${boundingBox.topLeft.y.toInt()} '
      '${boundingBox.width.toInt()}x${boundingBox.height.toInt()})';
}

/// Axis-aligned rectangle with normalized coordinates `[0.0, 1.0]`.
///
/// Values are expressed as fractions of the original image dimensions.
/// Utilities are provided to scale and expand the rectangle.
class RectF {
  /// Minimum X and Y plus maximum X and Y extents.
  final double xmin, ymin, xmax, ymax;

  /// Creates a normalized rectangle given its minimum and maximum extents.
  const RectF(this.xmin, this.ymin, this.xmax, this.ymax);

  /// Rectangle width.
  double get w => xmax - xmin;

  /// Rectangle height.
  double get h => ymax - ymin;

  /// Returns a rectangle scaled independently in X and Y.
  RectF scale(double sx, double sy) =>
      RectF(xmin * sx, ymin * sy, xmax * sx, ymax * sy);

  /// Expands the rectangle by [frac] in all directions, keeping the same center.
  RectF expand(double frac) {
    final double cx = (xmin + xmax) * 0.5;
    final double cy = (ymin + ymax) * 0.5;
    final double hw = (w * (1.0 + frac)) * 0.5;
    final double hh = (h * (1.0 + frac)) * 0.5;
    return RectF(cx - hw, cy - hh, cx + hw, cy + hh);
  }

  /// Converts this rect to a map for isolate serialization.
  Map<String, dynamic> toMap() => {
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax,
      };

  /// Creates a rect from a map.
  factory RectF.fromMap(Map<String, dynamic> map) => RectF(
        (map['xmin'] as num).toDouble(),
        (map['ymin'] as num).toDouble(),
        (map['xmax'] as num).toDouble(),
        (map['ymax'] as num).toDouble(),
      );
}

/// Raw detection output: bounding box + class index + score in normalized
/// `[0.0, 1.0]` coordinates.
///
/// Most users should use [DetectedObject] instead, which converts to absolute
/// pixel coordinates and resolves the category label.
class Detection {
  /// Normalized bounding box for the detection.
  final RectF boundingBox;

  /// Confidence score in `[0.0, 1.0]`.
  final double score;

  /// Class index into the label map.
  final int classIndex;

  /// Original image dimensions used to denormalize the box.
  final Size? imageSize;

  /// Creates a detection with normalized geometry, score, class index, and
  /// optional source size.
  Detection({
    required this.boundingBox,
    required this.score,
    required this.classIndex,
    this.imageSize,
  });

  /// Converts this detection to a map for isolate serialization.
  Map<String, dynamic> toMap() => {
        'boundingBox': boundingBox.toMap(),
        'score': score,
        'classIndex': classIndex,
        if (imageSize != null)
          'imageSize': {'width': imageSize!.width, 'height': imageSize!.height},
      };

  /// Creates a detection from a map.
  factory Detection.fromMap(Map<String, dynamic> map) => Detection(
        boundingBox:
            RectF.fromMap(Map<String, dynamic>.from(map['boundingBox'] as Map)),
        score: (map['score'] as num).toDouble(),
        classIndex: map['classIndex'] as int,
        imageSize: map['imageSize'] != null
            ? Size(
                (map['imageSize']['width'] as num).toDouble(),
                (map['imageSize']['height'] as num).toDouble(),
              )
            : null,
      );
}

/// Image tensor plus padding metadata used to undo letterboxing.
class ImageTensor {
  /// NHWC float tensor normalized to the model's expected range.
  final Float32List tensorNHWC;

  /// Padding fractions `[top, bottom, left, right]` applied during resize.
  final List<double> padding;

  /// Target width and height passed to the model.
  final int width, height;

  /// Creates an image tensor paired with the padding used during resize.
  ImageTensor(this.tensorNHWC, this.padding, this.width, this.height);
}

const _modelNameLite0 = 'efficientdet_lite0.tflite';
const _modelNameLite2 = 'efficientdet_lite2.tflite';
const _labelMapAsset = 'labelmap.txt';

/// The label map (90 entries, ordered by COCO class ID; some entries are
/// "???" placeholders to keep ID alignment with the original COCO IDs).
const int kLabelCount = 90;

/// EfficientDet-Lite0 input size (square).
const int kEfficientDetLite0Size = 320;

/// EfficientDet-Lite2 input size (square).
const int kEfficientDetLite2Size = 448;
