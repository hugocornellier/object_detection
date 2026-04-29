import 'package:flutter_test/flutter_test.dart';
import 'package:object_detection/object_detection.dart';

void main() {
  test('ObjectDetectionDart.registerWith does not throw', () {
    expect(ObjectDetectionDart.registerWith, isNotNull);
    expect(() => ObjectDetectionDart.registerWith(), returnsNormally);
  });
}
