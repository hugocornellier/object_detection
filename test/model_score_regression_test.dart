import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:object_detection/object_detection.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const variants = <ObjectDetectionModel, String>{
    ObjectDetectionModel.efficientDetLite0:
        'assets/models/efficientdet_lite0.tflite',
    ObjectDetectionModel.efficientDetLite2:
        'assets/models/efficientdet_lite2.tflite',
  };

  for (final entry in variants.entries) {
    test('${entry.key.name} keeps model probabilities below the threshold',
        () async {
      final modelBytes = await File(entry.value).readAsBytes();
      final model = await ObjectDetection.createFromBuffer(
        modelBytes,
        entry.key,
        performanceConfig: const PerformanceConfig(),
      );

      try {
        final elementCount = model.inputWidth * model.inputHeight * 3;
        for (final fill in <double>[-1.0, 0.0, 1.0]) {
          final input = Float32List(elementCount)
            ..fillRange(0, elementCount, fill);
          final detections = await model.callWithTensor(
            ImageTensor(
              input,
              const <double>[0.0, 0.0, 0.0, 0.0],
              model.inputWidth,
              model.inputHeight,
            ),
            scoreThreshold: 0.5,
          );

          expect(
            detections,
            isEmpty,
            reason: 'A uniform $fill input should not produce a >= 0.5 '
                'detection. Applying sigmoid twice turns every positive model '
                'probability into a false positive above 0.5.',
          );
        }
      } finally {
        model.dispose();
      }
    });
  }
}
