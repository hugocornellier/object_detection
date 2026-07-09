// Shared video-processing pipeline used by both the Video File screen and the
// batch demo-generation harness (integration_test/generate_demos_test.dart).
//
// Keeping the detection loop, temporal smoother, and OpenCV drawing in one
// place guarantees the annotated MP4s the app produces are byte-for-byte the
// same style the harness produces for README demos.

import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart' show Color;
import 'package:flutter_litert/flutter_litert.dart' show OneEuroFilter;
import 'package:object_detection/object_detection.dart';
import 'package:opencv_dart/opencv.dart' as cv;

/// Converts a Flutter [Color] to an OpenCV BGR scalar (alpha ignored).
cv.Scalar _bgr(Color c) => cv.Scalar(
      (c.b * 255).roundToDouble(),
      (c.g * 255).roundToDouble(),
      (c.r * 255).roundToDouble(),
    );

/// Box stroke thickness derived from frame resolution so boxes read the same
/// at 480p and 4K. Scaled by [thicknessScale] (a user multiplier, default 1).
int resolutionThickness(int w, int h, {double thicknessScale = 1.0}) {
  final int base = math.max(w, h);
  return math.max(2, (base / 540.0 * thicknessScale).round());
}

/// Label font scale derived from frame resolution, clamped so labels stay
/// legible without dominating the frame.
double _resolutionFontScale(int w, int h) {
  final int base = math.max(w, h);
  return (base / 1400.0).clamp(0.5, 1.6);
}

/// Draws bounding boxes + labels onto [mat] with OpenCV, mirroring what
/// [DetectionsPainter] renders on screen. Box thickness and label size scale
/// with the frame resolution (see [resolutionThickness]).
void drawObjectsOnMat(
  cv.Mat mat,
  List<DetectedObject> dets, {
  bool showBoxes = true,
  bool showLabels = true,
  bool perClassColors = true,
  Color boxColor = const Color(0xFF00FFCC),
  double thicknessScale = 1.0,
}) {
  if (dets.isEmpty) return;
  final int w = mat.cols;
  final int h = mat.rows;
  final int thickness =
      resolutionThickness(w, h, thicknessScale: thicknessScale);
  final double fontScale = _resolutionFontScale(w, h);
  final int fontThickness = math.max(1, (thickness * 0.55).round());
  final white = cv.Scalar(255, 255, 255);

  for (final obj in dets) {
    final color =
        _bgr(perClassColors ? colorForClass(obj.category.index) : boxColor);
    final bb = obj.boundingBox;
    final l = bb.topLeft.x.toInt().clamp(0, w - 1);
    final t = bb.topLeft.y.toInt().clamp(0, h - 1);
    final r = bb.bottomRight.x.toInt().clamp(0, w - 1);
    final b = bb.bottomRight.y.toInt().clamp(0, h - 1);

    if (showBoxes) {
      cv.rectangle(
        mat,
        cv.Rect(l, t, (r - l).clamp(1, w), (b - t).clamp(1, h)),
        color,
        thickness: thickness,
      );
    }

    if (showLabels) {
      final label =
          '${obj.categoryName} ${(obj.score * 100).toStringAsFixed(0)}%';
      final (sz, _) = cv.getTextSize(
          label, cv.FONT_HERSHEY_SIMPLEX, fontScale, fontThickness);
      final pad = (fontScale * 6).round();
      final labelTop = (t - sz.height - pad * 2).clamp(0, h - 1);
      final labelW = (sz.width + pad * 2).clamp(1, w - l);
      final labelH = (sz.height + pad * 2).clamp(1, h - labelTop);
      cv.rectangle(mat, cv.Rect(l, labelTop, labelW, labelH), color,
          thickness: -1);
      cv.putText(
        mat,
        label,
        cv.Point(l + pad, labelTop + sz.height + pad),
        cv.FONT_HERSHEY_SIMPLEX,
        fontScale,
        white,
        thickness: fontThickness,
      );
    }
  }
}

/// Runs the full annotate-a-video pipeline: reads [inputPath] frame-by-frame at
/// its native resolution and fps, detects objects, optionally smooths them with
/// [smoother], burns boxes onto each frame, and writes every frame to
/// [outputPath] (no frame dropping). Returns the number of frames written.
///
/// [onProgress] is called every few frames with (processed, total). [shouldCancel]
/// is polled each frame to allow early termination.
Future<({int frames, int width, int height, double fps, int total})>
    processVideoFile({
  required ObjectDetector detector,
  required String inputPath,
  required String outputPath,
  required ObjectDetectorOptions options,
  ObjectSmoother? smoother,
  bool showBoxes = true,
  bool showLabels = true,
  bool perClassColors = true,
  Color boxColor = const Color(0xFF00FFCC),
  double thicknessScale = 1.0,
  void Function(int processed, int total)? onProgress,
  bool Function()? shouldCancel,
}) async {
  final cap = cv.VideoCapture.fromFile(inputPath);
  if (!cap.isOpened) {
    cap.release();
    throw StateError('Could not open input video: $inputPath');
  }
  final fps = cap.get(cv.CAP_PROP_FPS);
  final width = cap.get(cv.CAP_PROP_FRAME_WIDTH).toInt();
  final height = cap.get(cv.CAP_PROP_FRAME_HEIGHT).toInt();
  final total = cap.get(cv.CAP_PROP_FRAME_COUNT).toInt();

  // Some OS video backends (macOS AVFoundation) refuse to open a writer over an
  // existing file, so remove any prior output to make regeneration idempotent.
  final outFile = File(outputPath);
  if (outFile.existsSync()) outFile.deleteSync();

  final writer =
      cv.VideoWriter.fromFile(outputPath, 'avc1', fps, (width, height));
  if (!writer.isOpened) {
    cap.release();
    writer.release();
    throw StateError('Could not open VideoWriter for $outputPath');
  }

  smoother?.reset();
  cv.Mat? frame;
  int idx = 0;
  try {
    while (!(shouldCancel?.call() ?? false)) {
      final result = cap.read(m: frame);
      final ok = result.$1;
      frame = result.$2;
      if (!ok || frame.isEmpty) break;

      final raw = await detector.detectFromMat(frame, options: options);
      final double tSec = fps > 0 ? idx / fps : idx / 30.0;
      final dets = smoother == null ? raw : smoother.apply(raw, tSec);
      drawObjectsOnMat(
        frame,
        dets,
        showBoxes: showBoxes,
        showLabels: showLabels,
        perClassColors: perClassColors,
        boxColor: boxColor,
        thicknessScale: thicknessScale,
      );
      writer.write(frame);
      idx++;
      if (idx % 4 == 0) onProgress?.call(idx, total);
    }
  } finally {
    cap.release();
    writer.release();
    frame?.dispose();
  }
  onProgress?.call(idx, total);
  return (frames: idx, width: width, height: height, fps: fps, total: total);
}

// ─────────────────────────── Object Smoother ──────────────────────────────

/// One tracked object: OneEuroFilters for its 4 box edges plus the last box
/// (used for IoU matching) and a missed-frame counter.
class _ObjectTrack {
  final List<OneEuroFilter> filters = [
    OneEuroFilter(minCutoff: 1.0, beta: 0.1, dCutoff: 1.0),
    OneEuroFilter(minCutoff: 1.0, beta: 0.1, dCutoff: 1.0),
    OneEuroFilter(minCutoff: 1.0, beta: 0.1, dCutoff: 1.0),
    OneEuroFilter(minCutoff: 1.0, beta: 0.1, dCutoff: 1.0),
  ];
  double lastLeft = 0, lastTop = 0, lastRight = 0, lastBottom = 0;
  int classIndex = -1;
  bool hasBox = false;
  int missedFrames = 0;
}

/// Temporally smooths object detections across video frames.
///
/// Matches this frame's detections to existing tracks by IoU (preferring the
/// same class), then runs each matched box's 4 edges through a [OneEuroFilter]
/// so boxes glide instead of jittering. Unmatched tracks age out after
/// [_maxMissed] frames. Mirrors the FaceSmoother used in face_detection_tflite,
/// adapted for objects (boxes only, no landmarks).
class ObjectSmoother {
  bool enabled;
  static const int _maxMissed = 5;
  static const double _minIou = 0.2;
  final List<_ObjectTrack> _tracks = [];

  ObjectSmoother({this.enabled = true});

  void reset() => _tracks.clear();

  List<DetectedObject> apply(List<DetectedObject> dets, double tSec) {
    if (!enabled || dets.isEmpty) {
      if (!enabled) _tracks.clear();
      return dets;
    }

    final unmatched = List<int>.generate(_tracks.length, (i) => i);
    final matched = List<int?>.filled(dets.length, null);

    for (int p = 0; p < dets.length; p++) {
      double bestIou = _minIou;
      int bestT = -1;
      for (final t in unmatched) {
        final track = _tracks[t];
        if (!track.hasBox) continue;
        final iou = _iou(dets[p], track);
        final sameClass = track.classIndex == dets[p].category.index;
        final effective = sameClass ? iou : iou * 0.5;
        if (effective > bestIou) {
          bestIou = effective;
          bestT = t;
        }
      }
      if (bestT >= 0) {
        matched[p] = bestT;
        unmatched.remove(bestT);
      }
    }

    final out = <DetectedObject>[];
    for (int p = 0; p < dets.length; p++) {
      _ObjectTrack track;
      if (matched[p] != null) {
        track = _tracks[matched[p]!];
        track.missedFrames = 0;
      } else {
        track = _ObjectTrack();
        _tracks.add(track);
      }
      out.add(_smooth(dets[p], track, tSec));
    }

    for (final t in unmatched) {
      _tracks[t].missedFrames++;
    }
    _tracks.removeWhere((t) => t.missedFrames > _maxMissed);

    return out;
  }

  DetectedObject _smooth(DetectedObject obj, _ObjectTrack track, double tSec) {
    final bb = obj.boundingBox;
    final double l = track.filters[0].filter(bb.topLeft.x, tSec);
    final double t = track.filters[1].filter(bb.topLeft.y, tSec);
    final double r = track.filters[2].filter(bb.bottomRight.x, tSec);
    final double b = track.filters[3].filter(bb.bottomRight.y, tSec);

    track.lastLeft = bb.topLeft.x;
    track.lastTop = bb.topLeft.y;
    track.lastRight = bb.bottomRight.x;
    track.lastBottom = bb.bottomRight.y;
    track.classIndex = obj.category.index;
    track.hasBox = true;

    final w = obj.originalSize.width;
    final h = obj.originalSize.height;
    final cat = obj.category;
    return DetectedObject(
      detection: Detection(
        boundingBox: RectF(l / w, t / h, r / w, b / h),
        score: cat.score,
        classIndex: cat.index,
      ),
      categories: obj.categories,
      originalSize: obj.originalSize,
    );
  }

  double _iou(DetectedObject a, _ObjectTrack b) {
    final box = a.boundingBox;
    final l = math.max(box.topLeft.x, b.lastLeft);
    final t = math.max(box.topLeft.y, b.lastTop);
    final r = math.min(box.bottomRight.x, b.lastRight);
    final bo = math.min(box.bottomRight.y, b.lastBottom);
    final iw = math.max(0.0, r - l);
    final ih = math.max(0.0, bo - t);
    final inter = iw * ih;
    final aa = math.max(0.0, box.bottomRight.x - box.topLeft.x) *
        math.max(0.0, box.bottomRight.y - box.topLeft.y);
    final bb = math.max(0.0, b.lastRight - b.lastLeft) *
        math.max(0.0, b.lastBottom - b.lastTop);
    final union = aa + bb - inter;
    if (union <= 0) return 0;
    return inter / union;
  }
}
