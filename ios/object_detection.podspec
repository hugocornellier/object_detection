#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint object_detection.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'object_detection'
  s.version          = '0.0.1'
  s.summary          = 'A new Flutter plugin project.'
  s.description      = <<-DESC
A new Flutter plugin project.
                       DESC
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }
  s.source           = { :path => '.' }
  s.source_files = 'object_detection/Sources/object_detection/**/*.{swift,h,m}'
  s.dependency 'Flutter'
  s.dependency 'TensorFlowLiteC', '~> 2.17.0'
  s.platform = :ios, '13.0'

  # Flutter.framework does not contain a i386 slice.
  # GCC_SYMBOLS_PRIVATE_EXTERN=NO ensures C symbols are exported for FFI lookup
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'GCC_SYMBOLS_PRIVATE_EXTERN' => 'NO'
  }
  s.swift_version = '5.0'

  s.resource_bundles = {'object_detection_privacy' => ['object_detection/Sources/object_detection/PrivacyInfo.xcprivacy']}
end
