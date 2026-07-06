// Detection-run analyzer.
//
// Runs the detector frame-by-frame over each input clip (same options as the
// demo harness) and reports continuous runs where EVERY frame has a detection.
// Because the smoother only draws a box on frames the detector actually hits
// (empty detections -> no box), a continuous detection run == a continuous
// drawn-box run, which is what a gap-free demo GIF needs.
//
// Usage:
//   flutter test integration_test/analyze_runs_test.dart -d macos \
//     --dart-define=INPUT_DIR=/abs/path/clips \
//     --dart-define=SCORE=60 --dart-define=MAX_RESULTS=3 \
//     --dart-define=CATEGORIES="sports ball"

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:object_detection/object_detection.dart';
import 'package:opencv_dart/opencv.dart' as cv;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('analyze detection runs', (tester) async {
    const inputDir = String.fromEnvironment('INPUT_DIR');
    const scorePct = int.fromEnvironment('SCORE', defaultValue: 60);
    const maxResults = int.fromEnvironment('MAX_RESULTS', defaultValue: 3);
    const categoriesCsv = String.fromEnvironment('CATEGORIES', defaultValue: '');
    final allowlist = categoriesCsv
        .split(',')
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList();
    final score = scorePct / 100.0;

    expect(inputDir.isNotEmpty, true, reason: 'INPUT_DIR define required');

    final detector = await ObjectDetector.create(
      model: ObjectDetectionModel.efficientDetLite2,
    );
    final options = ObjectDetectorOptions(
      scoreThreshold: score,
      maxResults: maxResults,
      categoryAllowlist: allowlist,
    );

    final inputs = Directory(inputDir)
        .listSync()
        .whereType<File>()
        .where((f) => f.path.toLowerCase().endsWith('.mp4'))
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));

    // ignore: avoid_print
    print('[runs] score=$score maxResults=$maxResults allowlist=$allowlist');

    for (final input in inputs) {
      final name = input.uri.pathSegments.last;
      final cap = cv.VideoCapture.fromFile(input.path);
      if (!cap.isOpened) {
        cap.release();
        // ignore: avoid_print
        print('[runs] $name  COULD NOT OPEN');
        continue;
      }
      final fps = cap.get(cv.CAP_PROP_FPS);
      final f = fps > 0 ? fps : 30.0;
      final presence = <bool>[];
      final scores = <double>[];
      cv.Mat? frame;
      while (true) {
        final res = cap.read(m: frame);
        final ok = res.$1;
        frame = res.$2;
        if (!ok || frame.isEmpty) break;
        final raw = await detector.detectFromMat(frame, options: options);
        presence.add(raw.isNotEmpty);
        scores.add(raw.isEmpty
            ? 0.0
            : raw.map((d) => d.score).reduce((a, b) => a > b ? a : b));
      }
      cap.release();
      frame.dispose();

      // Continuous runs of frames with >=1 detection.
      final runs = <List<int>>[]; // [startFrame, lengthFrames]
      int cur = 0, curStart = 0;
      for (int i = 0; i < presence.length; i++) {
        if (presence[i]) {
          if (cur == 0) curStart = i;
          cur++;
        } else {
          if (cur > 0) runs.add([curStart, cur]);
          cur = 0;
        }
      }
      if (cur > 0) runs.add([curStart, cur]);

      int bestLen = 0, bestStart = 0;
      for (final r in runs) {
        if (r[1] > bestLen) {
          bestLen = r[1];
          bestStart = r[0];
        }
      }
      final detFrames = presence.where((x) => x).length;
      // ignore: avoid_print
      print('[runs] $name  fps=${f.toStringAsFixed(1)} frames=${presence.length} '
          'detected=$detFrames (${(100 * detFrames / (presence.isEmpty ? 1 : presence.length)).toStringAsFixed(0)}%) '
          'LONGEST=${bestLen}f=${(bestLen / f).toStringAsFixed(2)}s '
          '@${(bestStart / f).toStringAsFixed(2)}s '
          '${bestLen / f >= 3.0 ? "<<< >=3s" : ""}');
      // Every run >= 1.5s, with its window and mean score.
      for (final r in runs) {
        final dur = r[1] / f;
        if (dur >= 1.5) {
          double sum = 0;
          for (int i = r[0]; i < r[0] + r[1]; i++) {
            sum += scores[i];
          }
          final mean = sum / r[1];
          // ignore: avoid_print
          print('    run ${(r[0] / f).toStringAsFixed(2)}s..${((r[0] + r[1]) / f).toStringAsFixed(2)}s '
              '(${dur.toStringAsFixed(2)}s, ${r[1]}f) meanScore=${(mean * 100).toStringAsFixed(0)}%');
        }
      }
    }
    await detector.dispose();
    // ignore: avoid_print
    print('[runs] DONE');
  }, timeout: const Timeout(Duration(minutes: 30)));
}
