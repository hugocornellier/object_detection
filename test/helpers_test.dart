import 'package:flutter_test/flutter_test.dart';
import 'package:object_detection/object_detection.dart';

void main() {
  group('parseLabelMap', () {
    test('skips empty lines and trims whitespace', () {
      const raw = '\nperson \n bicycle\n\n  car \n';
      final labels = parseLabelMap(raw);
      expect(labels, ['person', 'bicycle', 'car']);
    });

    test('preserves "???" placeholders', () {
      const raw = 'person\n???\ncar';
      final labels = parseLabelMap(raw);
      expect(labels, ['person', '???', 'car']);
    });
  });

  group('_applyOptions / testApplyOptions', () {
    Detection det(double s, int idx) => Detection(
          boundingBox: const RectF(0, 0, 0.1, 0.1),
          score: s,
          classIndex: idx,
        );

    test('drops below scoreThreshold and sorts by score desc', () {
      final dets = [det(0.4, 0), det(0.9, 1), det(0.7, 2), det(0.3, 3)];
      final out = testApplyOptions(
        dets,
        const ObjectDetectorOptions(scoreThreshold: 0.5),
        ['person', 'bicycle', 'car', 'motorcycle'],
      );
      expect(out.length, 2);
      expect(out[0].score, 0.9);
      expect(out[1].score, 0.7);
    });

    test('respects maxResults', () {
      final dets = [
        for (int i = 0; i < 5; i++) det(0.9 - i * 0.1, i),
      ];
      final out = testApplyOptions(
        dets,
        const ObjectDetectorOptions(scoreThreshold: 0.0, maxResults: 3),
        ['a', 'b', 'c', 'd', 'e'],
      );
      expect(out.length, 3);
    });

    test('honours categoryAllowlist', () {
      final dets = [det(0.9, 0), det(0.8, 1), det(0.7, 2)];
      final out = testApplyOptions(
        dets,
        const ObjectDetectorOptions(
          scoreThreshold: 0.0,
          categoryAllowlist: ['person', 'car'],
        ),
        ['person', 'bicycle', 'car'],
      );
      expect(out.length, 2);
      expect(out.map((d) => d.classIndex).toSet(), {0, 2});
    });

    test('honours categoryDenylist', () {
      final dets = [det(0.9, 0), det(0.8, 1), det(0.7, 2)];
      final out = testApplyOptions(
        dets,
        const ObjectDetectorOptions(
          scoreThreshold: 0.0,
          categoryDenylist: ['bicycle'],
        ),
        ['person', 'bicycle', 'car'],
      );
      expect(out.length, 2);
      expect(out.map((d) => d.classIndex).toSet(), {0, 2});
    });

    test('throws when both allowlist and denylist provided', () {
      final dets = [det(0.9, 0)];
      expect(
        () => testApplyOptions(
          dets,
          const ObjectDetectorOptions(
            categoryAllowlist: ['a'],
            categoryDenylist: ['b'],
          ),
          ['a', 'b'],
        ),
        throwsA(isA<ArgumentError>()),
      );
    });
  });

  group('_detectionLetterboxRemoval / testDetectionLetterboxRemoval', () {
    test('returns input unchanged for zero padding', () {
      final dets = [
        Detection(
          boundingBox: const RectF(0.2, 0.2, 0.8, 0.8),
          score: 0.9,
          classIndex: 0,
        ),
      ];
      final out = testDetectionLetterboxRemoval(dets, [0.0, 0.0, 0.0, 0.0]);
      expect(out.first.boundingBox.xmin, closeTo(0.2, 1e-9));
      expect(out.first.boundingBox.xmax, closeTo(0.8, 1e-9));
    });

    test('rescales after horizontal letterbox', () {
      // 20% padding on each horizontal side ⇒ valid x range [0.2, 0.8]
      final dets = [
        Detection(
          boundingBox: const RectF(0.2, 0.0, 0.8, 1.0),
          score: 0.9,
          classIndex: 0,
        ),
      ];
      final out = testDetectionLetterboxRemoval(dets, [0.0, 0.0, 0.2, 0.2]);
      // 0.2 → 0, 0.8 → 1 in unpadded space
      expect(out.first.boundingBox.xmin, closeTo(0.0, 1e-9));
      expect(out.first.boundingBox.xmax, closeTo(1.0, 1e-9));
      // y unchanged
      expect(out.first.boundingBox.ymin, closeTo(0.0, 1e-9));
      expect(out.first.boundingBox.ymax, closeTo(1.0, 1e-9));
    });
  });
}
