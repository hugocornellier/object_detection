// ignore_for_file: avoid_print

// Performance benchmark for ObjectDetector.
//
// Measures the detector end-to-end and per pipeline stage so regressions and
// improvements are attributable to a specific stage rather than a single
// opaque number. Emits one `BENCH_JSON` line per case for machine diffing.
//
// Run with:
//   flutter test integration_test/object_detector_benchmark_test.dart -d macos

import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:object_detection/object_detection.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

const int kIterations = 30;
const int kWarmup = 5;

const List<String> kSamples = [
  'assets/samples/street.jpg',
  'assets/samples/people.jpg',
  'assets/samples/kitchen.jpg',
  'assets/samples/dog.jpg',
  'assets/samples/cat.jpg',
];

/// Timing statistics for a single benchmark case.
class Stats {
  Stats(this.label, this.samplesUs);

  final String label;
  final List<int> samplesUs;

  double get meanMs =>
      samplesUs.reduce((a, b) => a + b) / samplesUs.length / 1000.0;

  double _pctMs(double p) {
    final sorted = List<int>.from(samplesUs)..sort();
    return sorted[((sorted.length - 1) * p).round()] / 1000.0;
  }

  double get p50Ms => _pctMs(0.50);
  double get p95Ms => _pctMs(0.95);
  double get minMs => samplesUs.reduce(math.min) / 1000.0;
  double get maxMs => samplesUs.reduce(math.max) / 1000.0;

  double get stdDevMs {
    final m = meanMs;
    final v = samplesUs
            .map((u) => math.pow(u / 1000.0 - m, 2).toDouble())
            .reduce((a, b) => a + b) /
        samplesUs.length;
    return math.sqrt(v);
  }

  Map<String, dynamic> toJson() => {
        'label': label,
        'n': samplesUs.length,
        'mean_ms': double.parse(meanMs.toStringAsFixed(4)),
        'p50_ms': double.parse(p50Ms.toStringAsFixed(4)),
        'p95_ms': double.parse(p95Ms.toStringAsFixed(4)),
        'min_ms': double.parse(minMs.toStringAsFixed(4)),
        'max_ms': double.parse(maxMs.toStringAsFixed(4)),
        'stddev_ms': double.parse(stdDevMs.toStringAsFixed(4)),
      };

  void emit() {
    print('BENCH_JSON ${jsonEncode(toJson())}');
    print(
      '  $label  mean=${meanMs.toStringAsFixed(3)}ms  '
      'p50=${p50Ms.toStringAsFixed(3)}  p95=${p95Ms.toStringAsFixed(3)}  '
      'min=${minMs.toStringAsFixed(3)}  max=${maxMs.toStringAsFixed(3)}  '
      'sd=${stdDevMs.toStringAsFixed(3)}',
    );
  }
}

/// Runs [body] [kWarmup] + [kIterations] times and returns the timed tail.
Future<Stats> timeAsync(String label, Future<void> Function() body) async {
  for (int i = 0; i < kWarmup; i++) {
    await body();
  }
  final samples = <int>[];
  final sw = Stopwatch();
  for (int i = 0; i < kIterations; i++) {
    sw
      ..reset()
      ..start();
    await body();
    sw.stop();
    samples.add(sw.elapsedMicroseconds);
  }
  return Stats(label, samples);
}

/// Synchronous sibling of [timeAsync].
Stats timeSync(String label, void Function() body) {
  for (int i = 0; i < kWarmup; i++) {
    body();
  }
  final samples = <int>[];
  final sw = Stopwatch();
  for (int i = 0; i < kIterations; i++) {
    sw
      ..reset()
      ..start();
    body();
    sw.stop();
    samples.add(sw.elapsedMicroseconds);
  }
  return Stats(label, samples);
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  for (final model in ObjectDetectionModel.values) {
    final modelTag = model.name;

    group('[$modelTag] end-to-end detect()', () {
      late ObjectDetector detector;

      setUpAll(() async {
        detector = await ObjectDetector.create(model: model);
      });

      tearDownAll(() async => detector.dispose());

      for (final path in kSamples) {
        final name = path.split('/').last;
        testWidgets('e2e $name', (_) async {
          final bytes = (await rootBundle.load(path)).buffer.asUint8List();
          final stats = await timeAsync(
            '$modelTag/e2e/$name',
            () async => detector.detect(bytes),
          );
          final dets = await detector.detect(bytes);
          print('  ($name -> ${dets.length} detections)');
          stats.emit();
        });
      }
    });

    group('[$modelTag] pipeline stages', () {
      late ObjectDetection od;

      setUpAll(() async {
        od = await ObjectDetection.create(model);
      });

      tearDownAll(() => od.dispose());

      for (final path in kSamples) {
        final name = path.split('/').last;
        testWidgets('stages $name', (_) async {
          final bytes = (await rootBundle.load(path)).buffer.asUint8List();

          // Stage 1: JPEG decode.
          final decodeStats = timeSync('$modelTag/decode/$name', () {
            cv.imdecode(bytes, cv.IMREAD_COLOR).dispose();
          });

          final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

          // Stage 2: letterbox + normalize into the model input tensor.
          final preStats = timeSync('$modelTag/preprocess/$name', () {
            convertImageToTensor(
              mat,
              outW: od.inputWidth,
              outH: od.inputHeight,
            );
          });

          // Stage 3: invoke + anchor decode + NMS.
          final pack = convertImageToTensor(
            mat,
            outW: od.inputWidth,
            outH: od.inputHeight,
          );
          final modelStats = await timeAsync('$modelTag/model/$name', () async {
            await od.callWithTensor(pack, scoreThreshold: 0.5);
          });

          mat.dispose();

          decodeStats.emit();
          preStats.emit();
          modelStats.emit();
        });
      }
    });
  }
}
