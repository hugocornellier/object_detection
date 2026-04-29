// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "object_detection",
    platforms: [
        .iOS("13.0")
    ],
    products: [
        .library(name: "object-detection", targets: ["object_detection"])
    ],
    dependencies: [],
    targets: [
        .target(
            name: "object_detection",
            dependencies: [],
            resources: [
                .process("PrivacyInfo.xcprivacy"),
            ]
        )
    ]
)
