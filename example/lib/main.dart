import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:object_detection/object_detection.dart';

void main() {
  runApp(const ObjectDetectionApp());
}

class ObjectDetectionApp extends StatelessWidget {
  const ObjectDetectionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'object_detection demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const _DemoHome(),
    );
  }
}

class _DemoHome extends StatefulWidget {
  const _DemoHome();

  @override
  State<_DemoHome> createState() => _DemoHomeState();
}

class _DemoHomeState extends State<_DemoHome> {
  ObjectDetector? _detector;
  ObjectDetectionModel _modelChoice = ObjectDetectionModel.efficientDetLite0;
  double _scoreThreshold = 0.5;
  int _maxResults = 10;

  Uint8List? _imageBytes;
  Size? _imageSize;
  List<DetectedObject> _detections = const [];
  int _lastInferenceMs = 0;
  bool _busy = false;
  String? _error;

  static const _samples = <(String, String)>[
    ('Cat & TV', 'assets/samples/cat.jpg'),
    ('Dog', 'assets/samples/dog.jpg'),
    ('People', 'assets/samples/people.jpg'),
    ('Street', 'assets/samples/street.jpg'),
    ('Kitchen', 'assets/samples/kitchen.jpg'),
  ];

  @override
  void initState() {
    super.initState();
    _initDetector();
  }

  Future<void> _initDetector() async {
    setState(() => _busy = true);
    try {
      final detector = await ObjectDetector.create(model: _modelChoice);
      if (!mounted) return;
      setState(() {
        _detector = detector;
        _busy = false;
      });
      await _loadSample(_samples.first.$2);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Init failed: $e';
        _busy = false;
      });
    }
  }

  Future<void> _switchModel(ObjectDetectionModel m) async {
    final old = _detector;
    setState(() {
      _detector = null;
      _busy = true;
      _modelChoice = m;
    });
    await old?.dispose();
    try {
      final detector = await ObjectDetector.create(model: m);
      if (!mounted) return;
      setState(() {
        _detector = detector;
        _busy = false;
      });
      if (_imageBytes != null) await _runDetection(_imageBytes!);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Switch model failed: $e';
        _busy = false;
      });
    }
  }

  Future<void> _loadSample(String assetPath) async {
    final data = await rootBundle.load(assetPath);
    await _setBytes(data.buffer.asUint8List());
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;
    final bytes = await File(picked.path).readAsBytes();
    await _setBytes(bytes);
  }

  Future<void> _setBytes(Uint8List bytes) async {
    final ui = await decodeImageFromList(bytes);
    setState(() {
      _imageBytes = bytes;
      _imageSize = Size(ui.width.toDouble(), ui.height.toDouble());
      _detections = const [];
      _error = null;
    });
    await _runDetection(bytes);
  }

  Future<void> _runDetection(Uint8List bytes) async {
    final det = _detector;
    if (det == null) return;
    setState(() => _busy = true);
    final sw = Stopwatch()..start();
    try {
      final results = await det.detect(
        bytes,
        options: ObjectDetectorOptions(
          scoreThreshold: _scoreThreshold,
          maxResults: _maxResults,
        ),
      );
      sw.stop();
      if (!mounted) return;
      setState(() {
        _detections = results;
        _lastInferenceMs = sw.elapsedMilliseconds;
        _busy = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Detection failed: $e';
        _busy = false;
      });
    }
  }

  @override
  void dispose() {
    _detector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('object_detection demo'),
        actions: [
          if (_lastInferenceMs > 0)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              child: Center(
                child: TimingBadge(
                  totalMs: _lastInferenceMs,
                  detectionMs: _lastInferenceMs,
                ),
              ),
            ),
        ],
      ),
      body: Column(
        children: [
          _buildControls(),
          const Divider(height: 1),
          Expanded(child: _buildPreview()),
          if (_detections.isNotEmpty)
            SizedBox(
              height: 90,
              child: _buildDetectionList(),
            ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _busy ? null : _pickImage,
        tooltip: 'Pick image',
        child: const Icon(Icons.image),
      ),
    );
  }

  Widget _buildControls() {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: [
              for (final m in ObjectDetectionModel.values)
                ChoiceChip(
                  label: Text(_modelLabel(m)),
                  selected: _modelChoice == m,
                  onSelected:
                      _busy ? null : (sel) => sel ? _switchModel(m) : null,
                ),
            ],
          ),
          const SizedBox(height: 4),
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: [
              for (final s in _samples)
                ActionChip(
                  label: Text(s.$1),
                  onPressed: _busy ? null : () => _loadSample(s.$2),
                ),
            ],
          ),
          const SizedBox(height: 4),
          CompactSlider(
            label: 'Score',
            value: _scoreThreshold,
            min: 0.0,
            max: 1.0,
            onChanged: (v) {
              setState(() => _scoreThreshold = v);
              if (_imageBytes != null) _runDetection(_imageBytes!);
            },
          ),
          CompactSlider(
            label: 'Max',
            value: _maxResults.toDouble(),
            min: 1.0,
            max: 30.0,
            onChanged: (v) {
              setState(() => _maxResults = v.round());
              if (_imageBytes != null) _runDetection(_imageBytes!);
            },
          ),
          if (_error != null)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(
                _error!,
                style: const TextStyle(color: Colors.red, fontSize: 12),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildPreview() {
    final bytes = _imageBytes;
    final sz = _imageSize;
    if (bytes == null || sz == null) {
      return Center(
        child: _busy
            ? const CircularProgressIndicator()
            : const Text('Pick an image or tap a sample.'),
      );
    }
    return LayoutBuilder(
      builder: (context, constraints) {
        final fit = applyBoxFit(
          BoxFit.contain,
          sz,
          Size(constraints.maxWidth, constraints.maxHeight),
        );
        final imgRect = Alignment.center.inscribe(
          fit.destination,
          Offset.zero & Size(constraints.maxWidth, constraints.maxHeight),
        );
        return Stack(
          alignment: Alignment.center,
          children: [
            Image.memory(
              bytes,
              fit: BoxFit.contain,
              gaplessPlayback: true,
              width: constraints.maxWidth,
              height: constraints.maxHeight,
            ),
            Positioned.fill(
              child: CustomPaint(
                painter: DetectionsPainter(
                  detections: _detections,
                  imageRectOnCanvas: imgRect,
                  originalImageSize: sz,
                ),
              ),
            ),
            if (_busy)
              const Positioned(
                bottom: 16,
                right: 16,
                child: SizedBox(
                  width: 24,
                  height: 24,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
              ),
          ],
        );
      },
    );
  }

  Widget _buildDetectionList() {
    return ListView.separated(
      scrollDirection: Axis.horizontal,
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
      itemCount: _detections.length,
      separatorBuilder: (_, __) => const SizedBox(width: 8),
      itemBuilder: (context, i) {
        final d = _detections[i];
        final color = colorForClass(d.category.index);
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            color: color.withAlpha(40),
            border: Border.all(color: color, width: 1.5),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                d.categoryName,
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  color: color,
                ),
              ),
              Text(
                '${(d.score * 100).toStringAsFixed(1)}%',
                style: const TextStyle(fontSize: 12),
              ),
            ],
          ),
        );
      },
    );
  }

  String _modelLabel(ObjectDetectionModel m) {
    switch (m) {
      case ObjectDetectionModel.efficientDetLite0:
        return 'Lite0';
      case ObjectDetectionModel.efficientDetLite2:
        return 'Lite2';
    }
  }
}
