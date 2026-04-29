import 'dart:typed_data';

import 'package:opencv_dart/opencv_dart.dart' as cv;

/// Test helpers for integration tests.
class TestUtils {
  /// Returns a minimal 1×1 PNG image as bytes.
  static Uint8List createDummyImageBytes() => ImageGenerator.create1x1Png();
}

/// Helper to fabricate small images entirely in memory for negative-path tests.
class ImageGenerator {
  /// Creates a minimal 1×1 PNG.
  static Uint8List create1x1Png() {
    return Uint8List.fromList([
      0x89,
      0x50,
      0x4E,
      0x47,
      0x0D,
      0x0A,
      0x1A,
      0x0A,
      0x00,
      0x00,
      0x00,
      0x0D,
      0x49,
      0x48,
      0x44,
      0x52,
      0x00,
      0x00,
      0x00,
      0x01,
      0x00,
      0x00,
      0x00,
      0x01,
      0x08,
      0x06,
      0x00,
      0x00,
      0x00,
      0x1F,
      0x15,
      0xC4,
      0x89,
      0x00,
      0x00,
      0x00,
      0x0A,
      0x49,
      0x44,
      0x41,
      0x54,
      0x78,
      0x9C,
      0x63,
      0x00,
      0x01,
      0x00,
      0x00,
      0x05,
      0x00,
      0x01,
      0x0D,
      0x0A,
      0x2D,
      0xB4,
      0x00,
      0x00,
      0x00,
      0x00,
      0x49,
      0x45,
      0x4E,
      0x44,
      0xAE,
      0x42,
      0x60,
      0x82,
    ]);
  }

  /// Creates a solid-color cv.Mat (BGR, 3-channel).
  static cv.Mat createSolidMat(
    int width,
    int height, {
    int r = 128,
    int g = 128,
    int b = 128,
  }) {
    final mat = cv.Mat.zeros(height, width, cv.MatType.CV_8UC3);
    mat.setTo(cv.Scalar(b.toDouble(), g.toDouble(), r.toDouble(), 255));
    return mat;
  }
}
