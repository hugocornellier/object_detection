// ignore_for_file: avoid_print

// Attributes the `model` stage (the dominant cost in the detector) to raw
// TFLite invoke vs. the Dart-side anchor decode + NMS, by driving a bare
// Interpreter with the same tensors the package uses.
//
// This is the test that tells us whether Dart-side postprocessing work is
// worth optimizing at all, or whether the pipeline is invoke-bound.
//
//   flutter test integration_test/stage_attribution_test.dart -d macos

import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/native.dart'
    show Interpreter, InterpreterFactory, PerformanceConfig, TensorFloat32Views;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:object_detection/object_detection.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

const int kIterations = 30;
const int kWarmup = 5;

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
    '  $label  mean=${_meanMs(us).toStringAsFixed(3)}ms  '
    'p50=${_p50Ms(us).toStringAsFixed(3)}ms',
  );
}

List<int> _bench(void Function() body) {
  for (int i = 0; i < kWarmup; i++) {
    body();
  }
  final us = <int>[];
  final sw = Stopwatch();
  for (int i = 0; i < kIterations; i++) {
    sw
      ..reset()
      ..start();
    body();
    sw.stop();
    us.add(sw.elapsedMicroseconds);
  }
  return us;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  for (final model in ObjectDetectionModel.values) {
    final tag = model.name;

    testWidgets('[$tag] invoke vs decode vs nms', (_) async {
      final bytes = (await rootBundle.load('assets/samples/street.jpg'))
          .buffer
          .asUint8List();
      final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      final itp = await Interpreter.fromAsset(
        'packages/object_detection/assets/models/${testNameFor(model)}',
        options: InterpreterFactory.create(const PerformanceConfig()).$1,
      );
      itp.allocateTensors();

      final inShape = itp.getInputTensor(0).shape;
      final inH = inShape[1], inW = inShape[2];
      final views = TensorFloat32Views.capture(itp);

      final pack = convertImageToTensor(mat, outW: inW, outH: inH);

      // Discover the output binding the same way the package does.
      int boxesIdx = -1, classesIdx = -1, numAnchors = 0, numClasses = 0;
      final outs = itp.getOutputTensors();
      for (int i = 0; i < outs.length; i++) {
        final s = outs[i].shape;
        if (s.length == 3 && s[2] == 4) {
          boxesIdx = i;
          numAnchors = s[1];
        } else if (s.length == 3 && s[2] > 4) {
          classesIdx = i;
          numClasses = s[2];
        }
      }
      print('  anchors=$numAnchors classes=$numClasses input=${inW}x$inH');

      // Stage A: tensor upload + invoke only.
      final invokeUs = _bench(() {
        views.inputs[0].setAll(0, pack.tensorNHWC);
        itp.invoke();
      });

      views.inputs[0].setAll(0, pack.tensorNHWC);
      itp.invoke();
      final boxBuf = itp.getOutputTensor(boxesIdx).data.buffer.asFloat32List();
      final clsBuf =
          itp.getOutputTensor(classesIdx).data.buffer.asFloat32List();

      final anchors = generateEfficientDetAnchors(imageSize: inW);

      // Stage B (old): nested-list anchors, unseeded argmax, one Detection
      // object allocated per survivor. This is the shape the package shipped
      // before the flat-anchor rewrite; kept here so old vs new are measured
      // back to back in one process, immune to cross-run drift.
      late List<Detection> decoded;
      final decodeOldUs = _bench(() {
        final out = <Detection>[];
        for (int i = 0; i < numAnchors; i++) {
          final classBase = i * numClasses;
          double best = -double.infinity;
          int bestCls = -1;
          for (int c = 0; c < numClasses; c++) {
            final v = clsBuf[classBase + c];
            if (v > best) {
              best = v;
              bestCls = c;
            }
          }
          if (best < 0.5) continue;
          final a = anchors[i];
          final boxBase = i * 4;
          final cy = boxBuf[boxBase] * a[3] + a[1];
          final cx = boxBuf[boxBase + 1] * a[2] + a[0];
          final h = math.exp(boxBuf[boxBase + 2]) * a[3];
          final w = math.exp(boxBuf[boxBase + 3]) * a[2];
          out.add(
            Detection(
              boundingBox:
                  RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
              score: best,
              classIndex: bestCls,
            ),
          );
        }
        decoded = out;
      });
      print('  survivors at 0.5 = ${decoded.length}');

      // Stage B (new): flat anchors, threshold-seeded argmax, results written
      // into preallocated typed scratch buffers.
      final flatAnchors = generateEfficientDetAnchorsFlat(imageSize: inW);
      final outBoxes = Float32List(2048 * 4);
      final outScores = Float32List(2048);
      final outCls = Int32List(2048);
      int newCount = 0;
      final decodeNewUs = _bench(() {
        int count = 0;
        for (int i = 0; i < numAnchors; i++) {
          final classBase = i * numClasses;
          final classEnd = classBase + numClasses;
          double best = 0.5;
          int bestCls = -1;
          for (int p = classBase; p < classEnd; p++) {
            if (clsBuf[p] >= best) {
              final v = clsBuf[p];
              if (bestCls < 0 || v > best) {
                best = v;
                bestCls = p - classBase;
              }
            }
          }
          if (bestCls < 0) continue;
          final aBase = i * 4;
          final cxA = flatAnchors[aBase];
          final cyA = flatAnchors[aBase + 1];
          final wA = flatAnchors[aBase + 2];
          final hA = flatAnchors[aBase + 3];
          final boxBase = i * 4;
          final cy = boxBuf[boxBase] * hA + cyA;
          final cx = boxBuf[boxBase + 1] * wA + cxA;
          final h = math.exp(boxBuf[boxBase + 2]) * hA;
          final w = math.exp(boxBuf[boxBase + 3]) * wA;
          final o = count * 4;
          outBoxes[o] = cx - w / 2;
          outBoxes[o + 1] = cy - h / 2;
          outBoxes[o + 2] = cx + w / 2;
          outBoxes[o + 3] = cy + h / 2;
          outScores[count] = best;
          outCls[count] = bestCls;
          count++;
        }
        newCount = count;
      });
      if (newCount != decoded.length) {
        throw StateError(
          'decode A/B disagreed: old kept ${decoded.length}, new kept '
          '$newCount',
        );
      }

      // Stage C: the List<List<double>> marshalling + weighted NMS.
      final nmsUs = _bench(() {
        final boxes = decoded
            .map((d) => [
                  d.boundingBox.xmin,
                  d.boundingBox.ymin,
                  d.boundingBox.xmax,
                  d.boundingBox.ymax,
                ])
            .toList();
        final scores = decoded.map((d) => d.score).toList();
        weightedNms(boxes, scores, iouThres: 0.45, maxDet: 200);
      });

      // Stage D: preprocessing, old (scalar Dart LUT loop, fresh allocation
      // every call) vs new (OpenCV SIMD convertTo into a reused buffer).
      final totalPixels = inW * inH;
      final preBuffer = Float32List(totalPixels * 3);
      final lbp = computeLetterboxParams(
        srcWidth: mat.cols,
        srcHeight: mat.rows,
        targetWidth: inW,
        targetHeight: inH,
      );
      final resized = cv.resize(
        mat,
        (lbp.newWidth, lbp.newHeight),
        interpolation: cv.INTER_LINEAR,
      );
      final padded = cv.copyMakeBorder(
        resized,
        lbp.padTop,
        lbp.padBottom,
        lbp.padLeft,
        lbp.padRight,
        cv.BORDER_CONSTANT,
        value: cv.Scalar.black,
      );
      resized.dispose();

      final preOldUs = _bench(() {
        bgrBytesToSignedFloat32(
          bytes: padded.data,
          totalPixels: totalPixels,
        );
      });
      final preNewUs = _bench(() {
        bgrMatToSignedFloat32(
          padded,
          totalPixels: totalPixels,
          buffer: preBuffer,
        );
      });

      // Equivalence check: the SIMD path must reproduce the scalar path.
      final scalarRef = bgrBytesToSignedFloat32(
        bytes: padded.data,
        totalPixels: totalPixels,
      );
      final simdRef = bgrMatToSignedFloat32(padded, totalPixels: totalPixels);
      double maxAbsDiff = 0.0;
      for (int i = 0; i < scalarRef.length; i++) {
        final d = (scalarRef[i] - simdRef[i]).abs();
        if (d > maxAbsDiff) maxAbsDiff = d;
      }
      print('  preprocess SIMD-vs-scalar max abs diff = $maxAbsDiff');
      padded.dispose();

      _emit('$tag/attr/invoke', invokeUs);
      _emit('$tag/attr/decode_old', decodeOldUs);
      _emit('$tag/attr/decode_new', decodeNewUs);
      _emit('$tag/attr/preprocess_old', preOldUs);
      _emit('$tag/attr/preprocess_new', preNewUs);
      _emit('$tag/attr/nms', nmsUs);

      mat.dispose();
      itp.close();
    });
  }
}
