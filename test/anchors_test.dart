import 'package:flutter_test/flutter_test.dart';
import 'package:object_detection/object_detection.dart';

void main() {
  group('generateEfficientDetAnchorsFlat', () {
    test('produces the anchor count each model expects', () {
      expect(
        generateEfficientDetAnchorsFlat(imageSize: kEfficientDetLite0Size)
                .length ~/
            4,
        19206,
      );
      expect(
        generateEfficientDetAnchorsFlat(imageSize: kEfficientDetLite2Size)
                .length ~/
            4,
        37629,
      );
    });

    test('is element-for-element identical to the nested-list generator', () {
      for (final size in [kEfficientDetLite0Size, kEfficientDetLite2Size]) {
        final flat = generateEfficientDetAnchorsFlat(imageSize: size);
        final nested = generateEfficientDetAnchors(imageSize: size);
        expect(nested.length, flat.length ~/ 4, reason: 'size $size');
        for (int i = 0; i < nested.length; i++) {
          for (int j = 0; j < 4; j++) {
            expect(
              nested[i][j],
              flat[i * 4 + j],
              reason: 'size $size, anchor $i, component $j',
            );
          }
        }
      }
    });

    test('emits normalized centers and positive extents', () {
      final flat =
          generateEfficientDetAnchorsFlat(imageSize: kEfficientDetLite0Size);
      for (int i = 0; i < flat.length; i += 4) {
        expect(flat[i], inInclusiveRange(0.0, 1.0), reason: 'cx at anchor $i');
        expect(
          flat[i + 1],
          inInclusiveRange(0.0, 1.0),
          reason: 'cy at anchor $i',
        );
        expect(flat[i + 2], greaterThan(0.0), reason: 'w at anchor $i');
        expect(flat[i + 3], greaterThan(0.0), reason: 'h at anchor $i');
      }
    });

    test('anchor box area grows with pyramid level', () {
      // P3 anchors (stride 8) come first and must be smaller than the P7
      // anchors (stride 128) that come last.
      final flat =
          generateEfficientDetAnchorsFlat(imageSize: kEfficientDetLite0Size);
      final firstArea = flat[2] * flat[3];
      final lastArea = flat[flat.length - 2] * flat[flat.length - 1];
      expect(lastArea, greaterThan(firstArea));
    });
  });
}
