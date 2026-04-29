import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:object_detection/object_detection.dart';

void main() {
  group('ObjectDetectorOptions', () {
    test('defaults are sane', () {
      const o = ObjectDetectorOptions.defaults;
      expect(o.scoreThreshold, 0.5);
      expect(o.maxResults, isNull);
      expect(o.categoryAllowlist, isEmpty);
      expect(o.categoryDenylist, isEmpty);
    });

    test('toMap/fromMap round-trip', () {
      const o = ObjectDetectorOptions(
        scoreThreshold: 0.3,
        maxResults: 5,
        categoryAllowlist: ['person', 'car'],
      );
      final back = ObjectDetectorOptions.fromMap(o.toMap());
      expect(back.scoreThreshold, 0.3);
      expect(back.maxResults, 5);
      expect(back.categoryAllowlist, ['person', 'car']);
      expect(back.categoryDenylist, isEmpty);
    });
  });

  group('Category', () {
    test('toMap/fromMap round-trip', () {
      const c = Category(
        index: 5,
        score: 0.87,
        categoryName: 'cat',
        displayName: 'cat',
      );
      final back = Category.fromMap(c.toMap());
      expect(back.index, 5);
      expect(back.score, closeTo(0.87, 1e-9));
      expect(back.categoryName, 'cat');
      expect(back.displayName, 'cat');
    });
  });

  group('RectF', () {
    test('expand grows symmetrically', () {
      const r = RectF(0.4, 0.4, 0.6, 0.6);
      final e = r.expand(1.0); // double width/height
      expect(e.w, closeTo(0.4, 1e-9));
      expect(e.h, closeTo(0.4, 1e-9));
      expect((e.xmin + e.xmax) / 2, closeTo(0.5, 1e-9));
      expect((e.ymin + e.ymax) / 2, closeTo(0.5, 1e-9));
    });

    test('toMap/fromMap round-trip', () {
      const r = RectF(0.1, 0.2, 0.3, 0.4);
      final back = RectF.fromMap(r.toMap());
      expect(back.xmin, 0.1);
      expect(back.ymin, 0.2);
      expect(back.xmax, 0.3);
      expect(back.ymax, 0.4);
    });
  });

  group('Detection', () {
    test('toMap/fromMap round-trip preserves imageSize', () {
      final d = Detection(
        boundingBox: const RectF(0.0, 0.0, 0.5, 0.5),
        score: 0.9,
        classIndex: 1,
        imageSize: const Size(100, 200),
      );
      final back = Detection.fromMap(d.toMap());
      expect(back.score, 0.9);
      expect(back.classIndex, 1);
      expect(back.imageSize?.width, 100);
      expect(back.imageSize?.height, 200);
    });
  });

  group('DetectedObject', () {
    test('boundingBox is in pixel coordinates', () {
      final det = Detection(
        boundingBox: const RectF(0.1, 0.2, 0.4, 0.5),
        score: 0.8,
        classIndex: 16,
      );
      final obj = DetectedObject(
        detection: det,
        categories: const [
          Category(
              index: 16, score: 0.8, categoryName: 'cat', displayName: 'cat'),
        ],
        originalSize: const Size(100, 100),
      );
      expect(obj.boundingBox.topLeft.x, closeTo(10.0, 1e-9));
      expect(obj.boundingBox.topLeft.y, closeTo(20.0, 1e-9));
      expect(obj.boundingBox.bottomRight.x, closeTo(40.0, 1e-9));
      expect(obj.boundingBox.bottomRight.y, closeTo(50.0, 1e-9));
      expect(obj.score, 0.8);
      expect(obj.categoryName, 'cat');
    });

    test('toMap/fromMap round-trip', () {
      final det = Detection(
        boundingBox: const RectF(0.1, 0.2, 0.4, 0.5),
        score: 0.8,
        classIndex: 16,
      );
      final obj = DetectedObject(
        detection: det,
        categories: const [
          Category(
              index: 16, score: 0.8, categoryName: 'cat', displayName: 'cat'),
        ],
        originalSize: const Size(100, 100),
      );
      final back = DetectedObject.fromMap(obj.toMap());
      expect(back.categoryName, 'cat');
      expect(back.boundingBox.topLeft.x, closeTo(10.0, 1e-9));
    });
  });

  group('ObjectDetectionModel', () {
    test('all variants map to a tflite filename', () {
      for (final m in ObjectDetectionModel.values) {
        final name = testNameFor(m);
        expect(name, endsWith('.tflite'));
      }
    });
  });
}
