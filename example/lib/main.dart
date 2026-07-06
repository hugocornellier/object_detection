import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:file_selector/file_selector.dart';
import 'package:flutter_colorpicker/flutter_colorpicker.dart';
import 'package:flutter_litert/flutter_litert.dart' show FrameThrottle;
import 'package:image_picker/image_picker.dart';
import 'package:object_detection/object_detection.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:video_player/video_player.dart';

import 'video_pipeline.dart';

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
      home: const HomeScreen(),
    );
  }
}

/// The default model used across all screens (higher accuracy; nicer boxes for
/// the demo). Live camera can switch to Lite0 for higher FPS.
const ObjectDetectionModel kDefaultModel = ObjectDetectionModel.efficientDetLite2;

String modelLabel(ObjectDetectionModel m) => switch (m) {
      ObjectDetectionModel.efficientDetLite0 => 'Lite0',
      ObjectDetectionModel.efficientDetLite2 => 'Lite2',
    };

/// Curated COCO classes exposed as one-tap filter chips (allowlist). Empty
/// selection means "all classes".
const List<String> kFilterClasses = [
  'person',
  'bicycle',
  'car',
  'motorcycle',
  'bus',
  'truck',
  'cat',
  'dog',
  'bird',
  'bottle',
  'chair',
  'laptop',
  'cell phone',
  'tv',
  'book',
];

// ─────────────────────────────── Home Menu ────────────────────────────────

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Object Detection Demo')),
      body: _ScrollableCentered(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 720),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: Text(
                    'Choose a Demo',
                    style: Theme.of(context).textTheme.headlineMedium,
                  ),
                ),
                const SizedBox(height: 28),
                _buildSection(
                  context,
                  'Object Detection',
                  [
                    _buildModeCard(
                      context,
                      icon: Icons.videocam,
                      title: 'Live Camera',
                      description: 'Real-time object detection from camera feed',
                      onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (_) => const LiveCameraScreen()),
                      ),
                    ),
                    _buildModeCard(
                      context,
                      icon: Icons.image,
                      title: 'Still Image',
                      description:
                          'Detect objects in photos from gallery or samples',
                      onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const StillImageScreen()),
                      ),
                    ),
                    _buildModeCard(
                      context,
                      icon: Icons.movie_creation_outlined,
                      title: 'Video File',
                      description:
                          'Process an MP4 frame-by-frame with smoothed, '
                          'tracked detections',
                      onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const VideoFileScreen()),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSection(BuildContext context, String title, List<Widget> cards) {
    final List<Widget> row = [];
    for (int i = 0; i < cards.length; i++) {
      if (i > 0) row.add(const SizedBox(width: 12));
      row.add(cards[i]);
    }
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Text(
            title,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                  color: Colors.grey[700],
                ),
          ),
        ),
        const SizedBox(height: 12),
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: IntrinsicHeight(
            child: Row(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: row,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildModeCard(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String description,
    required VoidCallback onTap,
  }) {
    return SizedBox(
      width: 190,
      child: Card(
        elevation: 4,
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(12),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(icon, size: 40, color: Colors.indigo),
                const SizedBox(height: 12),
                Text(title,
                    style: Theme.of(context).textTheme.titleMedium,
                    textAlign: TextAlign.center),
                const SizedBox(height: 6),
                Text(
                  description,
                  style: Theme.of(context)
                      .textTheme
                      .bodySmall
                      ?.copyWith(color: Colors.grey[600]),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ───────────────────────────── Still Image ────────────────────────────────

class StillImageScreen extends StatefulWidget {
  const StillImageScreen({super.key});
  @override
  State<StillImageScreen> createState() => _StillImageScreenState();
}

class _StillImageScreenState extends State<StillImageScreen> {
  ObjectDetector? _detector;
  ObjectDetectionModel _model = kDefaultModel;

  Uint8List? _imageBytes;
  Size? _originalSize;
  List<DetectedObject> _detections = const [];
  bool _isLoading = false;
  int? _detectionTimeMs;

  // Settings.
  bool _showBoundingBoxes = true;
  bool _showLabels = true;
  bool _perClassColors = true;
  Color _boundingBoxColor = const Color(0xFF00FFCC);
  double _boundingBoxThickness = 2.0;
  double _labelFontSize = 12.0;
  double _scoreThreshold = 0.5;
  int _maxResults = 15;
  final Set<String> _allowClasses = {};

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
    _initDetector(loadFirstSample: true);
  }

  Future<void> _initDetector({bool loadFirstSample = false}) async {
    try {
      await _detector?.dispose();
      _detector = await ObjectDetector.create(model: _model);
    } catch (_) {}
    if (!mounted) return;
    setState(() {});
    if (loadFirstSample) await _loadSample(_samples.first.$2);
  }

  ObjectDetectorOptions get _options => ObjectDetectorOptions(
        scoreThreshold: _scoreThreshold,
        maxResults: _maxResults,
        categoryAllowlist: _allowClasses.toList(),
      );

  @override
  void dispose() {
    _detector?.dispose();
    super.dispose();
  }

  Future<void> _pickAndRun() async {
    final picker = ImagePicker();
    final picked =
        await picker.pickImage(source: ImageSource.gallery, imageQuality: 100);
    if (picked == null) return;
    final bytes = await picked.readAsBytes();
    await _setBytes(bytes);
  }

  Future<void> _loadSample(String assetPath) async {
    final data = await rootBundle.load(assetPath);
    await _setBytes(data.buffer.asUint8List());
  }

  Future<void> _setBytes(Uint8List bytes) async {
    setState(() {
      _imageBytes = bytes;
      _detections = const [];
      _isLoading = true;
    });
    await _runDetection(bytes);
  }

  Future<void> _runDetection(Uint8List bytes) async {
    final det = _detector;
    if (det == null || !det.isReady) {
      setState(() => _isLoading = false);
      return;
    }
    setState(() => _isLoading = true);
    final sw = Stopwatch()..start();
    try {
      final results = await det.detect(bytes, options: _options);
      sw.stop();
      Size size;
      if (results.isNotEmpty) {
        size = results.first.originalSize;
      } else {
        final ui = await decodeImageFromList(bytes);
        size = Size(ui.width.toDouble(), ui.height.toDouble());
      }
      if (!mounted) return;
      setState(() {
        _detections = results;
        _originalSize = size;
        _detectionTimeMs = sw.elapsedMilliseconds;
        _isLoading = false;
      });
    } catch (_) {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  Future<void> _switchModel(ObjectDetectionModel m) async {
    setState(() {
      _model = m;
      _isLoading = true;
    });
    await _initDetector();
    if (_imageBytes != null) await _runDetection(_imageBytes!);
  }

  void _showSettingsSheet() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.6,
        minChildSize: 0.3,
        maxChildSize: 0.9,
        builder: (context, scrollController) => StatefulBuilder(
          builder: (context, setSheetState) {
            void update(VoidCallback fn, {bool rerun = false}) {
              fn();
              setSheetState(() {});
              setState(() {});
              if (rerun && _imageBytes != null) _runDetection(_imageBytes!);
            }

            Widget cb(String label, bool v, void Function(bool) set,
                    {bool rerun = false}) =>
                CompactCheckbox(
                    label: label,
                    value: v,
                    onChanged: (x) => update(() => set(x ?? false), rerun: rerun));
            Widget sl(String label, double v, double mn, double mx,
                    void Function(double) set,
                    {bool rerun = false}) =>
                CompactSlider(
                    label: label,
                    value: v,
                    min: mn,
                    max: mx,
                    onChanged: (x) => update(() => set(x), rerun: rerun));

            return Container(
              decoration: const BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
              ),
              child: Column(
                children: [
                  Container(
                    margin: const EdgeInsets.symmetric(vertical: 8),
                    width: 40,
                    height: 4,
                    decoration: BoxDecoration(
                      color: Colors.grey[300],
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                  Expanded(
                    child: ListView(
                      controller: scrollController,
                      padding: const EdgeInsets.symmetric(horizontal: 16),
                      children: [
                        ExpansionTile(
                          title: const Text('Display Options',
                              style: TextStyle(fontWeight: FontWeight.bold)),
                          initiallyExpanded: true,
                          children: [
                            Wrap(
                              spacing: 8,
                              runSpacing: 4,
                              children: [
                                cb('Bounding Boxes', _showBoundingBoxes,
                                    (v) => _showBoundingBoxes = v),
                                cb('Labels', _showLabels,
                                    (v) => _showLabels = v),
                                cb('Per-class colors', _perClassColors,
                                    (v) => _perClassColors = v),
                              ],
                            ),
                            const SizedBox(height: 8),
                          ],
                        ),
                        ExpansionTile(
                          title: const Text('Colors',
                              style: TextStyle(fontWeight: FontWeight.bold)),
                          children: [
                            Padding(
                              padding: const EdgeInsets.symmetric(vertical: 8),
                              child: _ColorPickerButton(
                                label: 'Box color (when per-class off)',
                                color: _boundingBoxColor,
                                onColorChanged: (c) =>
                                    update(() => _boundingBoxColor = c),
                              ),
                            ),
                          ],
                        ),
                        ExpansionTile(
                          title: const Text('Sizes',
                              style: TextStyle(fontWeight: FontWeight.bold)),
                          children: [
                            sl('Box thickness', _boundingBoxThickness, 0.5, 10.0,
                                (v) => _boundingBoxThickness = v),
                            sl('Label size', _labelFontSize, 8.0, 28.0,
                                (v) => _labelFontSize = v),
                            const SizedBox(height: 8),
                          ],
                        ),
                        ExpansionTile(
                          title: const Text('Detection',
                              style: TextStyle(fontWeight: FontWeight.bold)),
                          children: [
                            sl('Score threshold', _scoreThreshold, 0.0, 1.0,
                                (v) => _scoreThreshold = v, rerun: true),
                            sl('Max results', _maxResults.toDouble(), 1.0, 30.0,
                                (v) => _maxResults = v.round(), rerun: true),
                            const SizedBox(height: 8),
                          ],
                        ),
                        ExpansionTile(
                          title: const Text('Class filter',
                              style: TextStyle(fontWeight: FontWeight.bold)),
                          subtitle: Text(
                            _allowClasses.isEmpty
                                ? 'All classes'
                                : _allowClasses.join(', '),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                          children: [
                            Wrap(
                              spacing: 6,
                              runSpacing: 6,
                              children: [
                                for (final c in kFilterClasses)
                                  FilterChip(
                                    label: Text(c),
                                    selected: _allowClasses.contains(c),
                                    onSelected: (sel) => update(() {
                                      if (sel) {
                                        _allowClasses.add(c);
                                      } else {
                                        _allowClasses.remove(c);
                                      }
                                    }, rerun: true),
                                  ),
                              ],
                            ),
                            const SizedBox(height: 8),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final bool hasImage = _imageBytes != null && _originalSize != null;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Still Image Detection'),
        actions: [
          IconButton(
            onPressed: _isLoading ? null : _pickAndRun,
            icon: const Icon(Icons.add_photo_alternate),
            tooltip: 'Pick Image',
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4),
            child: DropdownButton<ObjectDetectionModel>(
              value: _model,
              underline: const SizedBox(),
              onChanged: _isLoading
                  ? null
                  : (v) {
                      if (v != null && v != _model) _switchModel(v);
                    },
              items: [
                for (final m in ObjectDetectionModel.values)
                  DropdownMenuItem(value: m, child: Text(modelLabel(m))),
              ],
            ),
          ),
          IconButton(
            onPressed: _showSettingsSheet,
            icon: const Icon(Icons.tune),
            tooltip: 'Settings',
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: Stack(
              children: [
                Center(
                  child: hasImage
                      ? LayoutBuilder(
                          builder: (context, constraints) {
                            final fitted = applyBoxFit(
                              BoxFit.contain,
                              _originalSize!,
                              Size(constraints.maxWidth, constraints.maxHeight),
                            );
                            final Size renderSize = fitted.destination;
                            final Rect imageRect = Alignment.center.inscribe(
                              renderSize,
                              Offset.zero &
                                  Size(constraints.maxWidth,
                                      constraints.maxHeight),
                            );
                            return Stack(
                              children: [
                                Positioned.fromRect(
                                  rect: imageRect,
                                  child: Image.memory(
                                    _imageBytes!,
                                    fit: BoxFit.fill,
                                    gaplessPlayback: true,
                                  ),
                                ),
                                Positioned.fromRect(
                                  rect: imageRect,
                                  child: CustomPaint(
                                    painter: DetectionsPainter(
                                      detections: _detections,
                                      imageRectOnCanvas: Rect.fromLTWH(
                                          0, 0, imageRect.width, imageRect.height),
                                      originalImageSize: _originalSize!,
                                      showBoundingBoxes: _showBoundingBoxes,
                                      showLabels: _showLabels,
                                      boundingBoxThickness: _boundingBoxThickness,
                                      labelFontSize: _labelFontSize,
                                      boundingBoxColor: _perClassColors
                                          ? null
                                          : _boundingBoxColor,
                                    ),
                                  ),
                                ),
                              ],
                            );
                          },
                        )
                      : _ScrollableCentered(
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(Icons.add_photo_alternate,
                                  size: 80, color: Colors.grey[300]),
                              const SizedBox(height: 16),
                              Text('No image selected',
                                  style: TextStyle(
                                      fontSize: 18, color: Colors.grey[600])),
                              const SizedBox(height: 8),
                              Text('Tap a sample below or pick an image',
                                  style: TextStyle(
                                      fontSize: 14, color: Colors.grey[500])),
                            ],
                          ),
                        ),
                ),
                if (hasImage && _detectionTimeMs != null)
                  Positioned(
                    top: 12,
                    left: 12,
                    child: TimingBadge(
                      totalMs: _detectionTimeMs!,
                      detectionMs: _detectionTimeMs,
                    ),
                  ),
                if (_isLoading)
                  Container(
                    color: Colors.black26,
                    child: const Center(child: CircularProgressIndicator()),
                  ),
              ],
            ),
          ),
          const Divider(height: 1),
          SizedBox(
            height: 56,
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
              itemCount: _samples.length,
              separatorBuilder: (_, __) => const SizedBox(width: 8),
              itemBuilder: (context, i) => ActionChip(
                label: Text(_samples[i].$1),
                onPressed:
                    _isLoading ? null : () => _loadSample(_samples[i].$2),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ───────────────────────────── Live Camera ────────────────────────────────

class LiveCameraScreen extends StatefulWidget {
  const LiveCameraScreen({super.key});
  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

class _LiveCameraScreenState extends State<LiveCameraScreen> {
  CameraController? _cameraController;
  List<CameraDescription> _availableCameras = const [];
  ObjectDetector? _detector;
  List<DetectedObject> _detections = const [];
  Size? _imageSize;
  int? _sensorOrientation;
  bool _isFrontCamera = false;
  bool _isSwitchingCamera = false;
  bool _isInitialized = false;
  bool _isImageStreamStarted = false;
  final FrameThrottle _throttle = FrameThrottle();
  final FpsCounter _fpsCounter = FpsCounter();
  int _fps = 0;
  int _detectionTimeMs = 0;

  DeviceOrientation _deviceOrientation = DeviceOrientation.portraitUp;
  StreamSubscription<AccelerometerEvent>? _accelerometerSub;

  ObjectDetectionModel _model = kDefaultModel;
  double _scoreThreshold = 0.4;
  bool _showLabels = true;

  @override
  void initState() {
    super.initState();
    _initCamera();
    if (!kIsWeb && (Platform.isAndroid || Platform.isIOS)) {
      _accelerometerSub = accelerometerEventStream().listen((event) {
        final next = event.x.abs() > event.y.abs()
            ? (event.x > 0
                ? DeviceOrientation.landscapeLeft
                : DeviceOrientation.landscapeRight)
            : (event.y > 0
                ? DeviceOrientation.portraitUp
                : DeviceOrientation.portraitDown);
        if (next == DeviceOrientation.portraitDown &&
            (_deviceOrientation == DeviceOrientation.landscapeLeft ||
                _deviceOrientation == DeviceOrientation.landscapeRight)) {
          return;
        }
        if (next != _deviceOrientation && mounted) {
          setState(() => _deviceOrientation = next);
        }
      });
    }
  }

  Future<void> _reinitDetector() async {
    final old = _detector;
    _detector = null;
    await old?.dispose();
    _detector = await ObjectDetector.create(model: _model);
  }

  Future<void> _initCamera() async {
    try {
      await _reinitDetector();

      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('No cameras available')),
          );
        }
        return;
      }
      _availableCameras = cameras;

      final camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      await _startControllerFor(camera, markInitialized: true);
    } catch (e, st) {
      debugPrint('Camera init failed: $e\n$st');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error initializing camera: $e')),
        );
      }
    }
  }

  Future<void> _startControllerFor(
    CameraDescription camera, {
    required bool markInitialized,
  }) async {
    final controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    await controller.initialize();
    if (!mounted) {
      await controller.dispose();
      return;
    }
    setState(() {
      _cameraController = controller;
      if (markInitialized) _isInitialized = true;
      _sensorOrientation = controller.description.sensorOrientation;
      _isFrontCamera =
          controller.description.lensDirection == CameraLensDirection.front;
    });
    await controller.startImageStream(_processCameraImage);
    _isImageStreamStarted = true;
  }

  bool get _canSwitchCamera {
    if (kIsWeb) return false;
    if (!(Platform.isAndroid || Platform.isIOS)) return false;
    final hasFront = _availableCameras
        .any((c) => c.lensDirection == CameraLensDirection.front);
    final hasBack = _availableCameras
        .any((c) => c.lensDirection == CameraLensDirection.back);
    return hasFront && hasBack;
  }

  Future<void> _switchCamera() async {
    if (_isSwitchingCamera || !_canSwitchCamera) return;
    final target =
        _isFrontCamera ? CameraLensDirection.back : CameraLensDirection.front;
    final next = _availableCameras.firstWhere(
      (c) => c.lensDirection == target,
      orElse: () => _availableCameras.first,
    );
    final prev = _cameraController;
    setState(() {
      _isSwitchingCamera = true;
      _cameraController = null;
      _detections = const [];
      _imageSize = null;
    });
    try {
      if (prev != null) {
        if (_isImageStreamStarted) {
          try {
            await prev.stopImageStream();
          } catch (_) {}
          _isImageStreamStarted = false;
        }
        await prev.dispose();
      }
      await _startControllerFor(next, markInitialized: false);
    } catch (e) {
      debugPrint('Camera switch failed: $e');
    } finally {
      if (mounted) setState(() => _isSwitchingCamera = false);
    }
  }

  Future<void> _switchModel(ObjectDetectionModel m) async {
    if (m == _model) return;
    setState(() => _model = m);
    await _reinitDetector();
  }

  DeviceOrientation _effectiveDeviceOrientation(BuildContext context) {
    final controller = _cameraController;
    if (controller != null) return controller.value.deviceOrientation;
    return MediaQuery.of(context).orientation == Orientation.portrait
        ? DeviceOrientation.portraitUp
        : DeviceOrientation.landscapeLeft;
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_fpsCounter.tick() && mounted) {
      setState(() => _fps = _fpsCounter.fps);
    }
    await _throttle.run(() async {
      try {
        final det = _detector;
        if (det == null || !det.isReady || !mounted) return;
        final sensor = _sensorOrientation;
        final CameraFrameRotation? rotation = sensor == null
            ? null
            : rotationForFrame(
                width: image.width,
                height: image.height,
                sensorOrientation: sensor,
                isFrontCamera: _isFrontCamera,
                deviceOrientation: _effectiveDeviceOrientation(context),
              );
        const int maxDim = 640;
        final Size size = detectionSize(
          width: image.width,
          height: image.height,
          rotation: rotation,
          maxDim: maxDim,
        );
        final sw = Stopwatch()..start();
        final results = await det.detectFromCameraImage(
          image,
          rotation: rotation,
          maxDim: maxDim,
          options: ObjectDetectorOptions(scoreThreshold: _scoreThreshold),
        );
        sw.stop();
        if (mounted) {
          setState(() {
            _detections = results;
            _imageSize = size;
            _detectionTimeMs = sw.elapsedMilliseconds;
          });
        }
      } catch (_) {
        // Keep the stream alive on transient errors.
      }
    });
  }

  @override
  void dispose() {
    _accelerometerSub?.cancel();
    if (_isImageStreamStarted) _cameraController?.stopImageStream();
    _cameraController?.dispose();
    _detector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized || _cameraController == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Live Camera Detection')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    final cameraAspectRatio = _cameraController!.value.aspectRatio;
    final effectiveOrientation = _effectiveDeviceOrientation(context);
    final bool isPortrait =
        effectiveOrientation == DeviceOrientation.portraitUp ||
            effectiveOrientation == DeviceOrientation.portraitDown;
    final double displayAspectRatio =
        isPortrait ? 1.0 / cameraAspectRatio : cameraAspectRatio;
    final int turns = barQuarterTurns(_deviceOrientation);
    final bool mirrorOverlay =
        (Platform.isAndroid && _isFrontCamera) || Platform.isWindows;

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          ObjectDetectionCameraOverlay(
            cameraPreview: CameraPreview(_cameraController!),
            displayAspectRatio: displayAspectRatio,
            mirrorHorizontally: mirrorOverlay,
            detections: _detections,
            imageSize: _imageSize,
            showLabels: _showLabels,
          ),
          _positionedTopBar(turns),
        ],
      ),
    );
  }

  Widget _positionedTopBar(int turns) {
    final bar = _buildCameraTopBar();
    final padding = MediaQuery.of(context).padding;
    if (turns == 0) {
      return Positioned(
          top: padding.top, left: padding.left, right: padding.right, child: bar);
    }
    return Positioned(
      top: padding.top,
      bottom: padding.bottom,
      left: turns == 3 ? padding.left : null,
      right: turns == 1 ? padding.right : null,
      width: kToolbarHeight,
      child: RotatedBox(quarterTurns: turns, child: bar),
    );
  }

  Widget _buildCameraTopBar() {
    final canPop = Navigator.of(context).canPop();
    return Material(
      color: Colors.black.withAlpha(179),
      elevation: 4,
      child: SizedBox(
        height: kToolbarHeight,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 4),
          child: Row(
            children: [
              if (canPop)
                IconButton(
                  tooltip: 'Back',
                  color: Colors.white,
                  icon: const Icon(Icons.arrow_back),
                  onPressed: () => Navigator.of(context).maybePop(),
                ),
              const SizedBox(width: 4),
              SizedBox(
                width: 66,
                child: Text('FPS: $_fps',
                    style: const TextStyle(color: Colors.white, fontSize: 14)),
              ),
              const Text(' | ',
                  style: TextStyle(color: Colors.white, fontSize: 14)),
              SizedBox(
                width: 64,
                child: Text('${_detectionTimeMs}ms',
                    style: const TextStyle(color: Colors.white, fontSize: 14)),
              ),
              const Spacer(),
              if (_canSwitchCamera)
                IconButton(
                  tooltip: 'Switch camera',
                  color: Colors.white,
                  icon: Icon(Platform.isIOS
                      ? Icons.flip_camera_ios
                      : Icons.flip_camera_android),
                  onPressed: _isSwitchingCamera ? null : _switchCamera,
                ),
              PopupMenuButton<void>(
                tooltip: 'Settings',
                icon: const Icon(Icons.settings, color: Colors.white),
                color: Colors.blueGrey[900],
                itemBuilder: (context) => [
                  PopupMenuItem<void>(
                    enabled: false,
                    padding: EdgeInsets.zero,
                    child: StatefulBuilder(
                      builder: (context, setMenuState) =>
                          _buildSettingsMenuContent(setMenuState),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSettingsMenuContent(StateSetter setMenuState) {
    void update(VoidCallback fn) {
      setState(fn);
      setMenuState(() {});
    }

    const sectionLabelStyle = TextStyle(
      color: Colors.white60,
      fontSize: 10,
      fontWeight: FontWeight.w600,
      letterSpacing: 1.2,
    );

    Widget chip(ObjectDetectionModel v, String label) {
      final selected = _model == v;
      return GestureDetector(
        onTap: () async {
          if (v == _model) return;
          await _switchModel(v);
          if (mounted) setMenuState(() {});
        },
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color: selected ? Colors.blue : Colors.white12,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Text(label,
              style: TextStyle(
                color: selected ? Colors.white : Colors.white70,
                fontSize: 12,
                fontWeight: selected ? FontWeight.bold : FontWeight.normal,
              )),
        ),
      );
    }

    return SizedBox(
      width: 260,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('MODEL', style: sectionLabelStyle),
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                chip(ObjectDetectionModel.efficientDetLite0, 'Lite0 (fast)'),
                chip(ObjectDetectionModel.efficientDetLite2, 'Lite2 (accurate)'),
              ],
            ),
            const Divider(color: Colors.white24, height: 24),
            const Text('SCORE THRESHOLD', style: sectionLabelStyle),
            Slider(
              value: _scoreThreshold,
              min: 0.0,
              max: 1.0,
              label: _scoreThreshold.toStringAsFixed(2),
              onChanged: (v) => update(() => _scoreThreshold = v),
            ),
            const Divider(color: Colors.white24, height: 8),
            Row(
              children: [
                const Expanded(
                  child: Text('Labels',
                      style: TextStyle(color: Colors.white70, fontSize: 14)),
                ),
                Switch(
                  value: _showLabels,
                  activeTrackColor: Colors.blue,
                  onChanged: (v) => update(() => _showLabels = v),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// ───────────────────────────── Video File ─────────────────────────────────

class VideoFileScreen extends StatefulWidget {
  const VideoFileScreen({super.key});
  @override
  State<VideoFileScreen> createState() => _VideoFileScreenState();
}

class _VideoFileScreenState extends State<VideoFileScreen> {
  ObjectDetector? _detector;
  ObjectDetectionModel _model = kDefaultModel;
  bool _isInitialized = false;
  bool _isProcessing = false;
  bool _cancelRequested = false;
  String? _errorMessage;
  String? _statusMessage;

  String? _outputPath;
  int _totalFrames = 0;
  int _processedFrames = 0;
  double _videoFps = 0;
  int _videoWidth = 0;
  int _videoHeight = 0;
  Duration _elapsed = Duration.zero;
  final Stopwatch _wallClock = Stopwatch();

  VideoPlayerController? _playerController;
  bool _playerReady = false;

  bool _smoothingEnabled = true;
  final ObjectSmoother _smoother = ObjectSmoother(enabled: true);

  // Paint options.
  bool _showBoundingBoxes = true;
  bool _showLabels = true;
  bool _perClassColors = true;
  Color _boundingBoxColor = const Color(0xFF00FFCC);
  // Multiplier on the resolution-derived box thickness (1.0 = default).
  double _boundingBoxThickness = 1.0;
  double _scoreThreshold = 0.4;
  int _maxResults = 20;

  bool get _supportsInAppPlayer {
    if (kIsWeb) return true;
    return Platform.isAndroid || Platform.isIOS || Platform.isMacOS;
  }

  @override
  void initState() {
    super.initState();
    _initDetector();
  }

  Future<void> _initDetector() async {
    try {
      final detector = await ObjectDetector.create(model: _model);
      if (!mounted) {
        await detector.dispose();
        return;
      }
      setState(() {
        _detector = detector;
        _isInitialized = true;
      });
    } catch (e) {
      if (mounted) setState(() => _errorMessage = 'Failed to init detector: $e');
    }
  }

  Future<void> _switchModel(ObjectDetectionModel m) async {
    if (m == _model || _isProcessing) return;
    setState(() {
      _model = m;
      _isInitialized = false;
    });
    final old = _detector;
    _detector = null;
    await old?.dispose();
    await _initDetector();
  }

  @override
  void dispose() {
    _cancelRequested = true;
    _detector?.dispose();
    _playerController?.dispose();
    super.dispose();
  }

  Future<void> _disposePlayer() async {
    final c = _playerController;
    _playerController = null;
    _playerReady = false;
    await c?.dispose();
  }

  Future<void> _initPlayerForOutput(String path) async {
    await _disposePlayer();
    if (!_supportsInAppPlayer) return;
    final controller = VideoPlayerController.file(File(path));
    _playerController = controller;
    try {
      await controller.initialize();
      await controller.setLooping(true);
      if (!mounted) {
        await controller.dispose();
        _playerController = null;
        return;
      }
      setState(() => _playerReady = true);
      await controller.play();
    } catch (e) {
      debugPrint('Could not load output video: $e');
    }
  }

  Future<void> _pickVideo() async {
    const typeGroup = XTypeGroup(
      label: 'Videos',
      extensions: ['mp4', 'mov', 'm4v'],
    );
    final XFile? file = await openFile(acceptedTypeGroups: [typeGroup]);
    if (file == null) return;
    await _processVideo(file.path);
  }

  Future<void> _processVideo(String path) async {
    final inputFile = File(path);
    if (!await inputFile.exists()) {
      setState(() => _errorMessage = 'File does not exist: $path');
      return;
    }

    final docs = await getApplicationDocumentsDirectory();
    final outName = 'object_${DateTime.now().millisecondsSinceEpoch}.mp4';
    final outPath = '${docs.path}/$outName';

    await _disposePlayer();
    setState(() {
      _outputPath = outPath;
      _processedFrames = 0;
      _totalFrames = 0;
      _isProcessing = true;
      _cancelRequested = false;
      _errorMessage = null;
      _statusMessage = 'Processing...';
      _elapsed = Duration.zero;
    });
    _wallClock
      ..reset()
      ..start();

    final options = ObjectDetectorOptions(
      scoreThreshold: _scoreThreshold,
      maxResults: _maxResults,
    );

    try {
      final result = await processVideoFile(
        detector: _detector!,
        inputPath: path,
        outputPath: outPath,
        options: options,
        smoother: _smoother,
        showBoxes: _showBoundingBoxes,
        showLabels: _showLabels,
        perClassColors: _perClassColors,
        boxColor: _boundingBoxColor,
        thicknessScale: _boundingBoxThickness,
        onProgress: (processed, total) {
          if (!mounted) return;
          setState(() {
            _processedFrames = processed;
            _totalFrames = total;
            _elapsed = _wallClock.elapsed;
          });
        },
        shouldCancel: () => !mounted || _cancelRequested,
      );
      if (mounted) {
        setState(() {
          _processedFrames = result.frames;
          _totalFrames = result.total;
          _videoFps = result.fps;
          _videoWidth = result.width;
          _videoHeight = result.height;
          _elapsed = _wallClock.elapsed;
          _statusMessage = _cancelRequested
              ? 'Cancelled after ${result.frames} frames.'
              : 'Done. Wrote ${result.frames} frames.';
        });
      }
    } catch (e) {
      String hint = '';
      if (Platform.isLinux) {
        hint = '\n\nLinux requires GStreamer plugins. Try:\n'
            '  sudo apt install gstreamer1.0-libav '
            'gstreamer1.0-plugins-good gstreamer1.0-plugins-bad';
      }
      if (mounted) {
        setState(() => _errorMessage =
            'Could not process video: $e\n\nThe format may be unsupported by '
                'the OS video backend, or the "avc1" (H.264) writer may be '
                'unavailable.$hint');
      }
    } finally {
      _wallClock.stop();
      if (mounted) setState(() => _isProcessing = false);
      if (mounted && !_cancelRequested && _outputPath != null) {
        await _initPlayerForOutput(_outputPath!);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Video File Detection'),
        actions: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4),
            child: DropdownButton<ObjectDetectionModel>(
              value: _model,
              underline: const SizedBox(),
              onChanged: _isProcessing
                  ? null
                  : (v) {
                      if (v != null) _switchModel(v);
                    },
              items: [
                for (final m in ObjectDetectionModel.values)
                  DropdownMenuItem(value: m, child: Text(modelLabel(m))),
              ],
            ),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          if (!_isInitialized)
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 24),
              child: Center(child: CircularProgressIndicator()),
            ),
          if (_errorMessage != null)
            Card(
              color: Colors.red[50],
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Text(_errorMessage!,
                    style: const TextStyle(color: Colors.red)),
              ),
            ),
          _buildSettingsCard(),
          const SizedBox(height: 12),
          if (!_isProcessing)
            ElevatedButton.icon(
              onPressed: _isInitialized ? _pickVideo : null,
              icon: const Icon(Icons.video_library),
              label: const Text('Pick a video (MP4)'),
            ),
          if (_isProcessing) _buildProgress(),
          const SizedBox(height: 16),
          if (_playerReady && _playerController != null && !_isProcessing)
            VideoResultCard(
              statusMessage: _statusMessage ?? 'Done',
              summary: _summaryText(),
              preview: _OutputVideoPlayer(controller: _playerController!),
            )
          else if (_statusMessage != null && !_isProcessing)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(_statusMessage!,
                        style: const TextStyle(fontWeight: FontWeight.w500)),
                    const SizedBox(height: 4),
                    Text(_summaryText()),
                    if (_outputPath != null) ...[
                      const SizedBox(height: 8),
                      SelectableText('Output: $_outputPath',
                          style: const TextStyle(fontSize: 12)),
                    ],
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }

  String _summaryText() {
    final secs = _elapsed.inMilliseconds / 1000.0;
    final procFps = secs > 0 ? (_processedFrames / secs) : 0.0;
    return '$_processedFrames frames  •  '
        '${_videoWidth}x$_videoHeight @ ${_videoFps.toStringAsFixed(0)} fps  •  '
        '${secs.toStringAsFixed(1)}s  •  '
        '${procFps.toStringAsFixed(1)} fps processing';
  }

  Widget _buildSettingsCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Options', style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 4),
            Wrap(
              spacing: 12,
              runSpacing: 4,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                CompactCheckbox(
                  label: 'Boxes',
                  value: _showBoundingBoxes,
                  onChanged: (v) =>
                      setState(() => _showBoundingBoxes = v ?? true),
                ),
                CompactCheckbox(
                  label: 'Labels',
                  value: _showLabels,
                  onChanged: (v) => setState(() => _showLabels = v ?? true),
                ),
                CompactCheckbox(
                  label: 'Per-class colors',
                  value: _perClassColors,
                  onChanged: (v) => setState(() => _perClassColors = v ?? true),
                ),
                CompactCheckbox(
                  label: 'Smoothing (tracking)',
                  value: _smoothingEnabled,
                  onChanged: (v) => setState(() {
                    _smoothingEnabled = v ?? true;
                    _smoother.enabled = _smoothingEnabled;
                  }),
                ),
                _ColorPickerButton(
                  label: 'Box color',
                  color: _boundingBoxColor,
                  onColorChanged: (c) => setState(() => _boundingBoxColor = c),
                ),
              ],
            ),
            CompactSlider(
              label: 'Box thickness x (scales with resolution)',
              value: _boundingBoxThickness,
              min: 0.5,
              max: 3.0,
              onChanged: (v) => setState(() => _boundingBoxThickness = v),
            ),
            CompactSlider(
              label: 'Score threshold',
              value: _scoreThreshold,
              min: 0.0,
              max: 1.0,
              onChanged: (v) => setState(() => _scoreThreshold = v),
            ),
            CompactSlider(
              label: 'Max results',
              value: _maxResults.toDouble(),
              min: 1.0,
              max: 30.0,
              onChanged: (v) => setState(() => _maxResults = v.round()),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildProgress() {
    final double pct =
        _totalFrames > 0 ? (_processedFrames / _totalFrames).clamp(0.0, 1.0) : 0;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text('Processing frame $_processedFrames'
                '${_totalFrames > 0 ? ' / $_totalFrames' : ''}'),
            const SizedBox(height: 8),
            LinearProgressIndicator(value: _totalFrames > 0 ? pct : null),
            const SizedBox(height: 12),
            Align(
              alignment: Alignment.centerRight,
              child: OutlinedButton.icon(
                onPressed: () => setState(() => _cancelRequested = true),
                icon: const Icon(Icons.stop),
                label: const Text('Cancel'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class VideoResultCard extends StatelessWidget {
  final String statusMessage;
  final String summary;
  final Widget preview;

  const VideoResultCard({
    super.key,
    required this.statusMessage,
    required this.summary,
    required this.preview,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.check_circle, color: Colors.green),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(statusMessage,
                      style: const TextStyle(fontWeight: FontWeight.w500)),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(summary),
            const SizedBox(height: 12),
            preview,
          ],
        ),
      ),
    );
  }
}

class VideoPlayerChrome extends StatelessWidget {
  final double aspectRatio;
  final Widget video;
  final Widget progress;
  final bool isPlaying;
  final String positionLabel;
  final VoidCallback onTogglePlay;

  const VideoPlayerChrome({
    super.key,
    required this.aspectRatio,
    required this.video,
    required this.progress,
    required this.isPlaying,
    required this.positionLabel,
    required this.onTogglePlay,
  });

  @override
  Widget build(BuildContext context) {
    final double maxPreviewHeight =
        math.max(120.0, MediaQuery.sizeOf(context).height * 0.45);
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Align(
          alignment: Alignment.centerLeft,
          child: ConstrainedBox(
            constraints: BoxConstraints(maxHeight: maxPreviewHeight),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: AspectRatio(
                aspectRatio: aspectRatio,
                child: Stack(
                  fit: StackFit.expand,
                  children: [Container(color: Colors.black), video],
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 8),
        LayoutBuilder(
          builder: (context, constraints) {
            final bool showTime = constraints.maxWidth >= 180;
            return Row(
              children: [
                IconButton(
                  icon: Icon(isPlaying ? Icons.pause : Icons.play_arrow),
                  onPressed: onTogglePlay,
                ),
                Expanded(child: progress),
                if (showTime) ...[
                  const SizedBox(width: 8),
                  Text(positionLabel),
                ],
              ],
            );
          },
        ),
      ],
    );
  }
}

class _OutputVideoPlayer extends StatefulWidget {
  final VideoPlayerController controller;
  const _OutputVideoPlayer({required this.controller});

  @override
  State<_OutputVideoPlayer> createState() => _OutputVideoPlayerState();
}

class _OutputVideoPlayerState extends State<_OutputVideoPlayer> {
  void _onTick() {
    if (mounted) setState(() {});
  }

  @override
  void initState() {
    super.initState();
    widget.controller.addListener(_onTick);
  }

  @override
  void didUpdateWidget(covariant _OutputVideoPlayer oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.controller != widget.controller) {
      oldWidget.controller.removeListener(_onTick);
      widget.controller.addListener(_onTick);
    }
  }

  @override
  void dispose() {
    widget.controller.removeListener(_onTick);
    super.dispose();
  }

  String _fmt(Duration d) {
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$m:$s';
  }

  @override
  Widget build(BuildContext context) {
    final c = widget.controller;
    final value = c.value;
    return VideoPlayerChrome(
      aspectRatio: value.aspectRatio == 0 ? 16 / 9 : value.aspectRatio,
      video: VideoPlayer(c),
      progress: VideoProgressIndicator(
        c,
        allowScrubbing: true,
        padding: const EdgeInsets.symmetric(vertical: 12),
      ),
      isPlaying: value.isPlaying,
      positionLabel: '${_fmt(value.position)} / ${_fmt(value.duration)}',
      onTogglePlay: () => value.isPlaying ? c.pause() : c.play(),
    );
  }
}

// ──────────────────────────── Shared widgets ──────────────────────────────

class _ScrollableCentered extends StatelessWidget {
  final Widget child;
  const _ScrollableCentered({required this.child});

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        return SingleChildScrollView(
          child: ConstrainedBox(
            constraints: BoxConstraints(minHeight: constraints.maxHeight),
            child: Center(child: child),
          ),
        );
      },
    );
  }
}

/// A small color swatch that opens a color picker dialog on tap.
class _ColorPickerButton extends StatelessWidget {
  final String label;
  final Color color;
  final ValueChanged<Color> onColorChanged;

  const _ColorPickerButton({
    required this.label,
    required this.color,
    required this.onColorChanged,
  });

  void _pick(BuildContext context) {
    Color temp = color;
    showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Pick $label'),
        content: SingleChildScrollView(
          child: ColorPicker(
            pickerColor: color,
            onColorChanged: (c) => temp = c,
            enableAlpha: false,
            portraitOnly: true,
          ),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          TextButton(
            onPressed: () {
              onColorChanged(temp);
              Navigator.pop(ctx);
            },
            child: const Text('Select'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: () => _pick(context),
      borderRadius: BorderRadius.circular(6),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 2),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 44,
              height: 26,
              decoration: BoxDecoration(
                color: color,
                borderRadius: BorderRadius.circular(6),
                border: Border.all(color: Colors.black26),
              ),
            ),
            const SizedBox(height: 4),
            Text(label, style: const TextStyle(fontSize: 11)),
          ],
        ),
      ),
    );
  }
}
