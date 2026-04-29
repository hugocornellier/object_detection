/// Object detection inference utilities backed by MediaPipe-style
/// TFLite models (EfficientDet-Lite, SSD MobileNetV2) for Flutter apps.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';

export 'src/dart_registration.dart';
export 'package:flutter_litert/flutter_litert.dart'
    show
        PerformanceMode,
        PerformanceConfig,
        createNHWCTensor4D,
        fillNHWC4D,
        allocTensorShape,
        flattenDynamicTensor,
        sigmoid,
        sigmoidClipped,
        clamp01,
        clip,
        computeLetterboxParams,
        LetterboxParams,
        bgrBytesToRgbFloat32,
        bgrBytesToSignedFloat32,
        Point,
        BoundingBox,
        packYuv420,
        YuvPlane,
        YuvLayout,
        PackedYuv,
        CameraPlane,
        CameraFrame,
        CameraFrameConversion,
        CameraFrameRotation,
        prepareCameraFrame,
        prepareCameraFrameFromImage,
        rotationForFrame,
        detectionSize,
        coverFitScaleOffset,
        barQuarterTurns,
        FpsCounter,
        drawLandmarkMarker,
        drawSkeletonConnections,
        drawBoundingBoxOutline,
        weightedNms;

export 'package:opencv_dart/opencv_dart.dart' show Mat, imdecode, IMREAD_COLOR;

part 'src/types_and_consts.dart';
part 'src/util/helpers.dart';
part 'src/object_detector.dart';
part 'src/isolate/object_detector_core.dart';
part 'src/models/object_detection_model.dart';
part 'src/ui/overlay_painters.dart';
part 'src/ui/timing_widgets.dart';
part 'src/ui/demo_controls.dart';
