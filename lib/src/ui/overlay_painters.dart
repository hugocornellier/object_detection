part of '../../object_detection.dart';

/// Classify detection-time in milliseconds into a display-friendly bucket
/// (`label`, `color`, `icon`) for overlay status indicators.
({String label, Color color, IconData icon}) performanceLevel(int ms) {
  if (ms < 200) {
    return (label: 'Excellent', color: Colors.green, icon: Icons.speed);
  } else if (ms < 500) {
    return (label: 'Good', color: Colors.lightGreen, icon: Icons.thumb_up);
  } else if (ms < 1000) {
    return (label: 'Fair', color: Colors.orange, icon: Icons.warning_amber);
  } else {
    return (label: 'Slow', color: Colors.red, icon: Icons.hourglass_bottom);
  }
}

/// Default per-class colors. The list cycles through 12 distinct hues so that
/// adjacent COCO classes get visually different overlay strokes.
const List<Color> kDetectionPalette = [
  Color(0xFFFF1744), // red
  Color(0xFF00E676), // green
  Color(0xFF2979FF), // blue
  Color(0xFFFFEA00), // yellow
  Color(0xFFD500F9), // magenta
  Color(0xFF00E5FF), // cyan
  Color(0xFFFF9100), // orange
  Color(0xFF76FF03), // lime
  Color(0xFFC51162), // pink
  Color(0xFF651FFF), // violet
  Color(0xFF1DE9B6), // teal
  Color(0xFFFFC400), // amber
];

/// Returns a deterministic color from [kDetectionPalette] for a given class
/// index. Useful for keeping the same color across frames.
Color colorForClass(int classIndex) =>
    kDetectionPalette[classIndex.abs() % kDetectionPalette.length];

/// Custom painter that renders detected object bounding boxes with class
/// labels and confidence scores onto a canvas, mapped from original image
/// coordinates to the rectangle the image is drawn in on screen.
class DetectionsPainter extends CustomPainter {
  /// Detected objects to render.
  final List<DetectedObject> detections;

  /// On-canvas rectangle the image is drawn in (e.g. from `applyBoxFit`).
  final Rect imageRectOnCanvas;

  /// Original image dimensions used to scale detection coordinates.
  final Size originalImageSize;

  /// Whether to draw bounding boxes.
  final bool showBoundingBoxes;

  /// Whether to draw class label + score text.
  final bool showLabels;

  /// Stroke thickness for boxes.
  final double boundingBoxThickness;

  /// Font size for labels.
  final double labelFontSize;

  /// Override color for boxes. If null, [colorForClass] is used per detection.
  final Color? boundingBoxColor;

  /// Background fill color for the label tag. Drawn at 70% opacity over the
  /// detection's class color.
  final Color labelBackground;

  /// Color for the label text.
  final Color labelTextColor;

  DetectionsPainter({
    required this.detections,
    required this.imageRectOnCanvas,
    required this.originalImageSize,
    this.showBoundingBoxes = true,
    this.showLabels = true,
    this.boundingBoxThickness = 2.0,
    this.labelFontSize = 12.0,
    this.boundingBoxColor,
    this.labelBackground = const Color(0xCC000000),
    this.labelTextColor = Colors.white,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (detections.isEmpty) return;

    final double ox = imageRectOnCanvas.left;
    final double oy = imageRectOnCanvas.top;
    final double scaleX = imageRectOnCanvas.width / originalImageSize.width;
    final double scaleY = imageRectOnCanvas.height / originalImageSize.height;

    for (final DetectedObject obj in detections) {
      final Color color = boundingBoxColor ?? colorForClass(obj.category.index);

      final BoundingBox bb = obj.boundingBox;
      final Rect rect = Rect.fromLTRB(
        ox + bb.topLeft.x * scaleX,
        oy + bb.topLeft.y * scaleY,
        ox + bb.bottomRight.x * scaleX,
        oy + bb.bottomRight.y * scaleY,
      );

      if (showBoundingBoxes) {
        final Paint boxPaint = Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = boundingBoxThickness
          ..color = color;
        canvas.drawRect(rect, boxPaint);
      }

      if (showLabels) {
        final String text =
            '${obj.categoryName} ${(obj.score * 100).toStringAsFixed(0)}%';
        final TextPainter tp = TextPainter(
          text: TextSpan(
            text: text,
            style: TextStyle(
              color: labelTextColor,
              fontSize: labelFontSize,
              fontWeight: FontWeight.bold,
            ),
          ),
          textDirection: TextDirection.ltr,
        );
        tp.layout();
        const double padX = 4.0;
        const double padY = 2.0;
        final double tagW = tp.width + padX * 2;
        final double tagH = tp.height + padY * 2;
        // Place label inside the box at the top-left, falling back below the box if it would clip.
        double tagLeft = rect.left;
        double tagTop = rect.top;
        if (tagTop < imageRectOnCanvas.top) tagTop = imageRectOnCanvas.top;
        if (tagLeft + tagW > imageRectOnCanvas.right) {
          tagLeft = imageRectOnCanvas.right - tagW;
        }
        final Rect tagRect = Rect.fromLTWH(tagLeft, tagTop, tagW, tagH);
        // Draw a colored bar matching the detection color over the dark base background.
        final Paint barPaint = Paint()
          ..color = color.withAlpha(220)
          ..style = PaintingStyle.fill;
        canvas.drawRect(tagRect, barPaint);
        tp.paint(canvas, Offset(tagLeft + padX, tagTop + padY));
      }
    }
  }

  @override
  bool shouldRepaint(covariant DetectionsPainter oldDelegate) =>
      oldDelegate.detections != detections ||
      oldDelegate.imageRectOnCanvas != imageRectOnCanvas ||
      oldDelegate.originalImageSize != originalImageSize ||
      oldDelegate.showBoundingBoxes != showBoundingBoxes ||
      oldDelegate.showLabels != showLabels;
}

/// Compute the axis-aligned bounding rect of a set of offsets.
Rect boundsOf(Iterable<Offset> pts) {
  final it = pts.iterator..moveNext();
  double minX = it.current.dx, maxX = it.current.dx;
  double minY = it.current.dy, maxY = it.current.dy;
  while (it.moveNext()) {
    final p = it.current;
    if (p.dx < minX) minX = p.dx;
    if (p.dx > maxX) maxX = p.dx;
    if (p.dy < minY) minY = p.dy;
    if (p.dy > maxY) maxY = p.dy;
  }
  return Rect.fromLTRB(minX, minY, maxX, maxY);
}
