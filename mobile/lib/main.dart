import 'package:flutter/material.dart';

import 'screens/home_screen.dart';
import 'settings.dart';
import 'theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final settings = AppSettings();
  await settings.load();
  runApp(LiftJudgeApp(settings: settings));
}

class LiftJudgeApp extends StatelessWidget {
  final AppSettings settings;
  const LiftJudgeApp({super.key, required this.settings});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Lift Judge',
        debugShowCheckedModeBanner: false,
        theme: buildTheme(),
        home: HomeScreen(settings: settings),
      );
}
