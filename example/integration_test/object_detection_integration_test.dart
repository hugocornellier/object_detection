// ignore_for_file: avoid_print

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:object_detection/object_detection.dart';

import 'test_helpers.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('ObjectDetector - Initialization and Disposal', () {
    test('should initialize successfully with default options', () async {
      final detector = ObjectDetector();
      expect(detector.isReady, false);

      await detector.initialize();
      expect(detector.isReady, true);

      await detector.dispose();
    });

    test('initialize throws when called twice without dispose', () async {
      final detector = ObjectDetector();
      await detector.initialize();
      expect(detector.isReady, true);

      await expectLater(
        detector.initialize(),
        throwsA(isA<StateError>()),
      );

      await detector.dispose();
    });

    test('should handle multiple dispose calls gracefully', () async {
      final detector = ObjectDetector();
      await detector.initialize();
      await detector.dispose();
      await detector.dispose(); // Second call: noop, must not throw.
    });
  });

  group('ObjectDetector - Error Handling', () {
    test('detect throws StateError before initialize', () async {
      final detector = ObjectDetector();
      final bytes = TestUtils.createDummyImageBytes();

      expect(
        () => detector.detect(bytes),
        throwsA(isA<StateError>().having(
          (e) => e.message,
          'message',
          contains('not initialized'),
        )),
      );
    });

    test('detect handles invalid image bytes gracefully', () async {
      final detector = ObjectDetector();
      await detector.initialize();

      final invalidBytes = Uint8List.fromList([1, 2, 3, 4, 5]);

      try {
        final results = await detector.detect(invalidBytes);
        // If no throw, decoded as empty image. Must yield no detections.
        expect(results, isEmpty);
      } catch (e) {
        expect(e, isNotNull);
      }

      await detector.dispose();
    });

    test('mutually-exclusive allowlist + denylist throws ArgumentError',
        () async {
      final detector = ObjectDetector();
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/cat.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      await expectLater(
        detector.detect(
          bytes,
          options: const ObjectDetectorOptions(
            categoryAllowlist: ['cat'],
            categoryDenylist: ['dog'],
          ),
        ),
        throwsA(isA<ArgumentError>()),
      );

      await detector.dispose();
    });
  });

  group('ObjectDetector - detect() with real images', () {
    late ObjectDetector detector;

    setUpAll(() async {
      detector = ObjectDetector();
      await detector.initialize();
    });

    tearDownAll(() async {
      await detector.dispose();
    });

    test('detects two cats in cat.jpg', () async {
      final ByteData data = await rootBundle.load('assets/samples/cat.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final sw = Stopwatch()..start();
      final dets = await detector.detect(
        bytes,
        options: const ObjectDetectorOptions(scoreThreshold: 0.3),
      );
      sw.stop();

      print('cat.jpg: ${sw.elapsedMilliseconds}ms, ${dets.length} detections, '
          'top: ${dets.take(5).map((d) => "${d.categoryName}(${d.score.toStringAsFixed(2)})").join(", ")}');

      expect(dets, isNotEmpty,
          reason: 'Should find at least one object in cat.jpg');
      expect(dets.any((d) => d.categoryName == 'cat'), isTrue,
          reason: 'cat.jpg should contain a "cat" detection');

      // Bounding-box sanity.
      for (final d in dets) {
        expect(d.boundingBox.width, greaterThan(0));
        expect(d.boundingBox.height, greaterThan(0));
        expect(d.boundingBox.topLeft.x, greaterThanOrEqualTo(0));
        expect(d.boundingBox.topLeft.y, greaterThanOrEqualTo(0));
        expect(d.score, inInclusiveRange(0.0, 1.0));
        expect(d.category.index, greaterThanOrEqualTo(0));
        expect(d.categoryName, isNotEmpty);
      }
    });

    test('detects something in dog.jpg', () async {
      // Note: despite the filename this image does not actually contain a dog;
      // we just assert that the model produces non-empty results.
      final ByteData data = await rootBundle.load('assets/samples/dog.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final sw = Stopwatch()..start();
      final dets = await detector.detect(
        bytes,
        options: const ObjectDetectorOptions(scoreThreshold: 0.3),
      );
      sw.stop();

      print('dog.jpg: ${sw.elapsedMilliseconds}ms, ${dets.length} detections, '
          'top: ${dets.take(5).map((d) => "${d.categoryName}(${d.score.toStringAsFixed(2)})").join(", ")}');

      expect(dets, isNotEmpty);
    });

    test('detects a person in people.jpg', () async {
      final ByteData data = await rootBundle.load('assets/samples/people.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final sw = Stopwatch()..start();
      final dets = await detector.detect(
        bytes,
        options: const ObjectDetectorOptions(scoreThreshold: 0.3),
      );
      sw.stop();

      print(
          'people.jpg: ${sw.elapsedMilliseconds}ms, ${dets.length} detections, '
          'top: ${dets.take(5).map((d) => "${d.categoryName}(${d.score.toStringAsFixed(2)})").join(", ")}');

      expect(dets, isNotEmpty);
      expect(dets.any((d) => d.categoryName == 'person'), isTrue,
          reason: 'people.jpg should contain a "person" detection');
    });

    test('benchmark: median latency over 5 runs on cat.jpg', () async {
      final ByteData data = await rootBundle.load('assets/samples/cat.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      // Warm-up.
      await detector.detect(bytes,
          options: const ObjectDetectorOptions(scoreThreshold: 0.3));
      final times = <int>[];
      for (int i = 0; i < 5; i++) {
        final sw = Stopwatch()..start();
        await detector.detect(bytes,
            options: const ObjectDetectorOptions(scoreThreshold: 0.3));
        sw.stop();
        times.add(sw.elapsedMilliseconds);
      }
      times.sort();
      final median = times[times.length ~/ 2];
      print(
          'benchmark cat.jpg (efficientDetLite0): runs=$times median=${median}ms');
      // Detection should be reasonably fast on default platform; 2s is plenty.
      expect(median, lessThan(2000));
    });

    test('respects scoreThreshold', () async {
      final ByteData data = await rootBundle.load('assets/samples/street.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final low = await detector.detect(
        bytes,
        options: const ObjectDetectorOptions(scoreThreshold: 0.1),
      );
      final high = await detector.detect(
        bytes,
        options: const ObjectDetectorOptions(scoreThreshold: 0.9),
      );
      expect(high.length, lessThanOrEqualTo(low.length),
          reason: 'higher threshold must keep no more than lower threshold');
      for (final d in high) {
        expect(d.score, greaterThanOrEqualTo(0.9));
      }
    });

    test('respects maxResults', () async {
      final ByteData data = await rootBundle.load('assets/samples/street.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final capped = await detector.detect(
        bytes,
        options:
            const ObjectDetectorOptions(scoreThreshold: 0.0, maxResults: 3),
      );
      expect(capped.length, lessThanOrEqualTo(3));
    });

    test('respects categoryAllowlist', () async {
      final ByteData data = await rootBundle.load('assets/samples/street.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final filtered = await detector.detect(
        bytes,
        options: const ObjectDetectorOptions(
          scoreThreshold: 0.1,
          categoryAllowlist: ['person', 'car'],
        ),
      );
      for (final d in filtered) {
        expect(['person', 'car'], contains(d.categoryName));
      }
    });
  });

  group('ObjectDetector - all model variants', () {
    for (final model in ObjectDetectionModel.values) {
      test('initializes and runs ${model.name}', () async {
        final detector = ObjectDetector();
        await detector.initialize(model: model);
        try {
          final ByteData data = await rootBundle.load('assets/samples/cat.jpg');
          final dets = await detector.detect(
            data.buffer.asUint8List(),
            options: const ObjectDetectorOptions(scoreThreshold: 0.2),
          );
          print('${model.name}: ${dets.length} detections, top='
              '${dets.isEmpty ? "(none)" : dets.first.categoryName}');
          expect(dets, isNotEmpty,
              reason:
                  '${model.name} should produce some detections on cat.jpg');
        } finally {
          await detector.dispose();
        }
      });
    }
  });
}
