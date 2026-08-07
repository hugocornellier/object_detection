// ignore_for_file: avoid_print

// Engine/delegate sweep for the EfficientDet-Lite graphs.
//
// Compares the Interpreter delegates (disabled / xnnpack / gpu / coreml)
// against the LiteRT Next CompiledModel engine (GPU-with-CPU-fallback and
// CPU-pinned) on identical input tensors, measuring pure inference.
//
// Runs on -d macos, which exercises the real Metal GPU and CoreML/ANE.
//
//   flutter test integration_test/engine_sweep_test.dart -d macos

import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/native.dart'
    show
        Accelerator,
        CompiledModel,
        Interpreter,
        InterpreterFactory,
        PerformanceConfig,
        PerformanceMode,
        Precision,
        TensorBufferMode,
        TensorFloat32Views,
        compiledModelFromBufferAuto;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:object_detection/object_detection.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

const int kIterations = 30;
const int kWarmup = 8;

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
    '  ${label.padRight(46)} p50=${_p50Ms(us).toStringAsFixed(3)}ms  '
    'mean=${_meanMs(us).toStringAsFixed(3)}ms',
  );
}

Future<List<int>> _benchAsync(Future<void> Function() body) async {
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

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  for (final model in ObjectDetectionModel.values) {
    final tag = model.name;

    testWidgets('[$tag] engine sweep', (_) async {
      final jpeg = (await rootBundle.load('assets/samples/street.jpg'))
          .buffer
          .asUint8List();
      final mat = cv.imdecode(jpeg, cv.IMREAD_COLOR);
      final modelBytes = (await rootBundle.load(
        'packages/object_detection/assets/models/${testNameFor(model)}',
      ))
          .buffer
          .asUint8List();

      final int side =
          model == ObjectDetectionModel.efficientDetLite0 ? 320 : 448;
      final pack = convertImageToTensor(mat, outW: side, outH: side);
      final Float32List input = pack.tensorNHWC;

      for (final mode in PerformanceMode.values) {
        try {
          final (opts, delegate) =
              InterpreterFactory.create(PerformanceConfig(mode: mode));
          final itp = Interpreter.fromBuffer(modelBytes, options: opts);
          itp.allocateTensors();
          final views = TensorFloat32Views.capture(itp);
          final us = await _benchAsync(() async {
            views.inputs[0].setAll(0, input);
            itp.invoke();
          });
          _emit('$tag/interp/${mode.name}', us);
          itp.close();
          delegate?.delete();
        } catch (e) {
          print('  $tag/interp/${mode.name}: FAILED ($e)');
        }
      }

      for (final entry in <String, Set<Accelerator>>{
        'gpu+cpu': {Accelerator.gpu, Accelerator.cpu},
        'cpu': {Accelerator.cpu},
      }.entries) {
        try {
          final CompiledModel compiled = compiledModelFromBufferAuto(
            modelBytes,
            accelerators: entry.value,
            precision: Precision.fp16,
            onGpuFallback: (e) => print('  (gpu compile fell back: $e)'),
          );
          print('  compiled accelerators = ${compiled.accelerators}');
          final us = await _benchAsync(() async {
            await compiled.runAsync([input]);
          });
          _emit('$tag/compiled/${entry.key}', us);
          compiled.close();
        } catch (e) {
          print('  $tag/compiled/${entry.key}: FAILED ($e)');
        }
      }

      // Synchronous run() vs runAsync(): the detector already owns a
      // background isolate, so runAsync's helper isolate is a pure extra hop.
      for (final tensorMode in TensorBufferMode.values) {
        try {
          final CompiledModel compiled =
              CompiledModel.fromBufferWithGpuFallback(
            modelBytes,
            precision: Precision.fp16,
            tensorBufferMode: tensorMode,
            onFallback: (e) => print('  (gpu compile fell back: $e)'),
          );
          print(
            '  ${tensorMode.name} compiled on ${compiled.accelerators}',
          );
          _emit(
            '$tag/compiled-sync-${tensorMode.name}/run',
            await _benchAsync(() async => compiled.run([input])),
          );

          if (tensorMode == TensorBufferMode.hostMemory) {
            // Zero-copy: write straight into the model's aligned host memory
            // and read outputs back as views, never materializing the ~1.8M
            // (Lite0) / 3.4M (Lite2) output floats as fresh Dart lists.
            _emit(
              '$tag/compiled-zerocopy/write+dispatch+read',
              await _benchAsync(() async {
                compiled.writeInput(0, (dst) => dst.setAll(0, input));
                compiled.dispatch();
                compiled.readOutput(0, (a) {
                  compiled.readOutput(1, (b) => a.length + b.length);
                });
              }),
            );
          }
          compiled.close();
        } catch (e) {
          print('  $tag/compiled-sync-${tensorMode.name}: FAILED ($e)');
        }
      }

      // Same comparison one level up, at ObjectDetection.callWithTensor, which
      // adds the anchor decode and NMS on top of inference. This is where the
      // compiled engine's output-copy cost shows up: EfficientDet emits
      // ~1.8M floats per frame (Lite0) and the interpreter path reads them as
      // a zero-copy view of native memory, while CompiledModel copies them
      // into fresh Dart lists on every call.
      final interpModel = await ObjectDetection.createFromBuffer(
        modelBytes,
        model,
        performanceConfig: const PerformanceConfig(),
      );
      _emit(
        '$tag/callWithTensor/interpreter',
        await _benchAsync(
          () async => interpModel.callWithTensor(pack, scoreThreshold: 0.5),
        ),
      );
      interpModel.dispose();

      final compiledModel =
          await ObjectDetection.createCompiledFromBuffer(modelBytes, model);
      print('  callWithTensor compiled on ${compiledModel.activeAccelerators}');
      _emit(
        '$tag/callWithTensor/compiled',
        await _benchAsync(
          () async => compiledModel.callWithTensor(pack, scoreThreshold: 0.5),
        ),
      );
      compiledModel.dispose();

      mat.dispose();
    });
  }
}
