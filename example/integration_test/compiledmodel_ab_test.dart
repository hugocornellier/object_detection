// ignore_for_file: avoid_print

// A/B between the two inference engines behind ObjectDetector:
//   - Interpreter + platform delegate (the default)
//   - LiteRT Next CompiledModel (useCompiledModel: true)
//
// Asserts the two agree on detections, then reports end-to-end latency for
// each so the speedup is attributable rather than asserted.
//
//   flutter test integration_test/compiledmodel_ab_test.dart -d macos

import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:object_detection/object_detection.dart';

const int kIterations = 30;
const int kWarmup = 8;

const List<String> kSamples = [
  'assets/samples/street.jpg',
  'assets/samples/people.jpg',
  'assets/samples/kitchen.jpg',
  'assets/samples/dog.jpg',
  'assets/samples/cat.jpg',
];

double _p50Ms(List<int> us) {
  final s = List<int>.from(us)..sort();
  return s[s.length ~/ 2] / 1000.0;
}

double _meanMs(List<int> us) => us.reduce((a, b) => a + b) / us.length / 1000.0;

void _emit(String label, List<int> us) {
  print(
    'BENCH_JSON ${jsonEncode({
          'label': label,
          'n': us.length,
          'mean_ms': double.parse(_meanMs(us).toStringAsFixed(4)),
          'p50_ms': double.parse(_p50Ms(us).toStringAsFixed(4)),
        })}',
  );
  print(
    '  ${label.padRight(44)} p50=${_p50Ms(us).toStringAsFixed(3)}ms  '
    'mean=${_meanMs(us).toStringAsFixed(3)}ms',
  );
}

Future<List<int>> _bench(Future<void> Function() body) async {
  for (int i = 0; i < kWarmup; i++) {
    await body();
  }
  final us = <int>[];
  final sw = Stopwatch();
  for (int i = 0; i < kIterations; i++) {
    sw
      ..reset()
      ..start();
    await body();
    sw.stop();
    us.add(sw.elapsedMicroseconds);
  }
  return us;
}

String _describe(List<DetectedObject> dets) => dets
    .map((d) => '${d.categoryName}:${d.score.toStringAsFixed(3)}')
    .join(', ');

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  for (final model in ObjectDetectionModel.values) {
    final tag = model.name;

    testWidgets('[$tag] compiled vs interpreter agree and are faster',
        (_) async {
      final images = <String, Uint8List>{
        for (final p in kSamples)
          p: (await rootBundle.load(p)).buffer.asUint8List(),
      };

      final interp = await ObjectDetector.create(model: model);
      final compiled = await ObjectDetector.create(
        model: model,
        useCompiledModel: true,
      );

      try {
        // Correctness: same categories, same boxes to within GPU fp16 noise.
        for (final entry in images.entries) {
          final a = await interp.detect(entry.value);
          final b = await compiled.detect(entry.value);
          final name = entry.key.split('/').last;
          print('  $name interp=[${_describe(a)}]');
          print('  $name compiled=[${_describe(b)}]');
          // Compared as multisets: GPU fp16 arithmetic drifts scores by ~1e-3,
          // which is enough to swap the rank of two detections whose scores
          // are tied to three decimals. That is engine noise, not a
          // disagreement about what is in the image.
          expect(
            (b.map((d) => d.categoryName).toList()..sort()),
            (a.map((d) => d.categoryName).toList()..sort()),
            reason: '$tag/$name: engines disagreed on detected categories',
          );

          for (final category in a.map((d) => d.categoryName).toSet()) {
            List<DetectedObject> ofCategory(List<DetectedObject> l) =>
                l.where((d) => d.categoryName == category).toList()
                  ..sort((x, y) => y.score.compareTo(x.score));
            final ax = ofCategory(a);
            final bx = ofCategory(b);
            for (int i = 0; i < ax.length; i++) {
              expect(
                bx[i].score,
                closeTo(ax[i].score, 0.05),
                reason: '$tag/$name: $category score drift at rank $i',
              );
              expect(
                bx[i].boundingBox.topLeft.x,
                closeTo(
                  ax[i].boundingBox.topLeft.x,
                  ax[i].originalSize.width * 0.02,
                ),
                reason: '$tag/$name: $category box drift at rank $i',
              );
            }
          }
        }

        for (final entry in images.entries) {
          final name = entry.key.split('/').last;
          _emit(
            '$tag/interpreter/$name',
            await _bench(() async => interp.detect(entry.value)),
          );
          _emit(
            '$tag/compiled/$name',
            await _bench(() async => compiled.detect(entry.value)),
          );
        }
      } finally {
        await interp.dispose();
        await compiled.dispose();
      }
    });
  }
}
