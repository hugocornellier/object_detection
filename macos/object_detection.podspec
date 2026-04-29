Pod::Spec.new do |s|
  s.name                  = 'object_detection'
  s.version               = '0.0.1'
  s.summary               = 'Object detection via TensorFlow Lite (macOS)'
  s.description           = 'Flutter plugin for on-device object detection using TensorFlow Lite.'
  s.homepage              = 'https://github.com/your/repo'
  s.license               = { :type => 'MIT' }
  s.authors               = { 'You' => 'you@example.com' }
  s.source                = { :path => '.' }

  s.platform              = :osx, '11.0'
  s.swift_version         = '5.0'

  s.source_files          = 'object_detection/Sources/object_detection/**/*.{swift,h,m}'

  s.resource_bundles      = { 'object_detection_privacy' => ['object_detection/Sources/object_detection/PrivacyInfo.xcprivacy'] }

  s.dependency            'FlutterMacOS'
  s.static_framework      = true

  # TFLite libraries are provided by flutter_litert dependency
end
