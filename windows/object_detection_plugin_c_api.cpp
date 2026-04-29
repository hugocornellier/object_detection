#include "include/object_detection/object_detection_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "object_detection_plugin.h"

void ObjectDetectionPluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  object_detection::ObjectDetectionPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
