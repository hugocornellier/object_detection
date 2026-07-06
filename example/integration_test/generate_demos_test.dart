// Batch demo generator.
//
// Runs the shared video pipeline (same code the Video File screen uses) over a
// folder of input clips, writing annotated, tracked, smoothed outputs. Runs as
// an integration test so it executes inside a real Flutter engine (rootBundle
// model assets, native OpenCV/LiteRT, and detection isolates all work).
//
// Usage:
//   flutter test integration_test/generate_demos_test.dart -d macos \
//     --dart-define=INPUT_DIR=/abs/path/demo_inputs \
//     --dart-define=OUTPUT_DIR=/abs/path/demo_outputs
//
// Optional defines: SCORE (default 0.35), MAX_RESULTS (default 25),
// THICKNESS (multiplier, default 1.0), SMOOTH (true/false, default true).

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:object_detection/object_detection.dart';
import 'package:object_detection_example/video_pipeline.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('generate annotated demo videos', (tester) async {
    const inputDir = String.fromEnvironment('INPUT_DIR');
    const outputDir = String.fromEnvironment('OUTPUT_DIR');
    const scorePct = int.fromEnvironment('SCORE', defaultValue: 35);
    const maxResults = int.fromEnvironment('MAX_RESULTS', defaultValue: 25);
    const thicknessPct = int.fromEnvironment('THICKNESS', defaultValue: 100);
    const smooth = bool.fromEnvironment('SMOOTH', defaultValue: true);
    // Comma-separated category allowlist, e.g. CATEGORIES="sports ball,person".
    // Empty means no category filtering.
    const categoriesCsv = String.fromEnvironment('CATEGORIES', defaultValue: '');
    final allowlist = categoriesCsv
        .split(',')
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList();
    final score = scorePct / 100.0;
    final thickness = thicknessPct / 100.0;

    expect(inputDir.isNotEmpty, true, reason: 'INPUT_DIR define required');
    expect(outputDir.isNotEmpty, true, reason: 'OUTPUT_DIR define required');

    Directory(outputDir).createSync(recursive: true);

    final inputs = Directory(inputDir)
        .listSync()
        .whereType<File>()
        .where((f) => f.path.toLowerCase().endsWith('.mp4'))
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));

    // ignore: avoid_print
    print('[demos] ${inputs.length} inputs, model=Lite2 score=$score '
        'maxResults=$maxResults thickness=${thickness}x smooth=$smooth '
        'allowlist=$allowlist');

    final detector = await ObjectDetector.create(
      model: ObjectDetectionModel.efficientDetLite2,
    );

    final options = ObjectDetectorOptions(
      scoreThreshold: score,
      maxResults: maxResults,
      categoryAllowlist: allowlist,
    );

    final overall = Stopwatch()..start();
    for (int i = 0; i < inputs.length; i++) {
      final input = inputs[i];
      final name = input.uri.pathSegments.last;
      final outPath = '$outputDir/$name';
      final sw = Stopwatch()..start();
      try {
        final res = await processVideoFile(
          detector: detector,
          inputPath: input.path,
          outputPath: outPath,
          options: options,
          smoother: smooth ? ObjectSmoother() : null,
          thicknessScale: thickness,
          onProgress: (p, total) {
            if (p % 60 == 0) {
              // ignore: avoid_print
              print('[demos] (${i + 1}/${inputs.length}) $name  $p/$total');
            }
          },
        );
        sw.stop();
        final procFps = sw.elapsedMilliseconds > 0
            ? res.frames * 1000 / sw.elapsedMilliseconds
            : 0.0;
        // ignore: avoid_print
        print('[demos] DONE (${i + 1}/${inputs.length}) $name  '
            '${res.width}x${res.height} frames=${res.frames} '
            '${(sw.elapsedMilliseconds / 1000).toStringAsFixed(1)}s '
            '${procFps.toStringAsFixed(1)}fps');
      } catch (e) {
        // ignore: avoid_print
        print('[demos] FAILED $name: $e');
      }
    }
    overall.stop();
    await detector.dispose();
    // ignore: avoid_print
    print('[demos] ALL DONE in ${(overall.elapsedMilliseconds / 1000).toStringAsFixed(1)}s '
        '-> $outputDir');
  }, timeout: const Timeout(Duration(minutes: 90)));
}
