#include "include/object_detection/object_detection_plugin.h"
#include "object_detection_plugin.h"
#include <flutter/plugin_registrar_windows.h>

void ObjectDetectionPluginRegisterWithRegistrar(FlutterDesktopPluginRegistrarRef registrar) {
  auto cpp_registrar =
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar);
  object_detection::ObjectDetectionPlugin::RegisterWithRegistrar(cpp_registrar);
}
