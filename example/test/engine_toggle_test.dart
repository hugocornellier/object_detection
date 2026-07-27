import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:object_detection_example/main.dart';

void main() {
  testWidgets('demo defaults to the CompiledModel engine', (tester) async {
    expect(
      kDefaultUseCompiledModel,
      isTrue,
      reason: 'The demo should showcase the faster LiteRT Next engine, '
          'matching the face/pose/hand demos.',
    );
  });

  testWidgets('badge names the engine, not a delegate', (tester) async {
    Future<void> pumpBadge(bool useCompiledModel) => tester.pumpWidget(
          MaterialApp(
            home: Scaffold(
              body: EngineToggleButton(
                useCompiledModel: useCompiledModel,
                onPressed: () {},
              ),
            ),
          ),
        );

    await pumpBadge(true);
    expect(find.text('CM'), findsOneWidget);
    expect(find.text('Interpreter'), findsNothing);

    await pumpBadge(false);
    expect(find.text('Interpreter'), findsOneWidget);
    expect(find.text('CM'), findsNothing);

    // "XNN" names a delegate the interpreter path only uses on some
    // platforms; on iOS it is Metal. The badge must not claim otherwise.
    expect(find.text('XNN'), findsNothing);
  });

  testWidgets('tapping the badge flips the engine', (tester) async {
    bool useCompiledModel = kDefaultUseCompiledModel;

    await tester.pumpWidget(
      MaterialApp(
        home: StatefulBuilder(
          builder: (context, setState) => Scaffold(
            body: EngineToggleButton(
              useCompiledModel: useCompiledModel,
              onPressed: () =>
                  setState(() => useCompiledModel = !useCompiledModel),
            ),
          ),
        ),
      ),
    );

    expect(find.text('CM'), findsOneWidget);
    await tester.tap(find.byType(EngineToggleButton));
    await tester.pump();
    expect(find.text('Interpreter'), findsOneWidget);
    expect(useCompiledModel, isFalse);

    await tester.tap(find.byType(EngineToggleButton));
    await tester.pump();
    expect(find.text('CM'), findsOneWidget);
    expect(useCompiledModel, isTrue);
  });

  testWidgets('a null callback disables the badge', (tester) async {
    // The screens pass null while a detector is being rebuilt, so the badge
    // must render disabled rather than queueing a second engine swap.
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: EngineToggleButton(useCompiledModel: true, onPressed: null),
        ),
      ),
    );
    expect(find.text('CM'), findsOneWidget);
    expect(
      tester.widget<TextButton>(find.byType(TextButton)).enabled,
      isFalse,
    );
  });
}
