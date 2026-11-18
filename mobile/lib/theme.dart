import 'package:flutter/material.dart';

import 'models.dart';

/// Competition-platform look: dark stage, white/red referee lights, IPF card colours.
class AppColors {
  static const stage = Color(0xFF0E0F12);
  static const panel = Color(0xFF181A1F);
  static const line = Color(0xFF2A2D34);
  static const chalk = Color(0xFFF2F0EB);
  static const muted = Color(0xFF9A9CA3);
  static const good = Color(0xFFF7F7F2); // white light
  static const noLift = Color(0xFFE5383B); // red light
  static const borderline = Color(0xFFF2A541);
  static const accent = Color(0xFFE5383B);

  static Color card(RefCard c) => switch (c) {
        RefCard.red => const Color(0xFFD62828),
        RefCard.blue => const Color(0xFF1F6FEB),
        RefCard.yellow => const Color(0xFFF4C430),
      };

  static Color status(CheckStatus s) => switch (s) {
        CheckStatus.pass => const Color(0xFF52B788),
        CheckStatus.fail => noLift,
        CheckStatus.borderline || CheckStatus.unknown => borderline,
      };
}

ThemeData buildTheme() {
  final base = ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    colorScheme: ColorScheme.fromSeed(
      seedColor: AppColors.accent,
      brightness: Brightness.dark,
      surface: AppColors.stage,
    ),
    scaffoldBackgroundColor: AppColors.stage,
  );
  return base.copyWith(
    appBarTheme: const AppBarTheme(
      backgroundColor: AppColors.stage,
      foregroundColor: AppColors.chalk,
      elevation: 0,
      centerTitle: false,
    ),
    cardTheme: CardThemeData(
      color: AppColors.panel,
      elevation: 0,
      margin: EdgeInsets.zero,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(6),
        side: const BorderSide(color: AppColors.line),
      ),
    ),
    textTheme: base.textTheme.apply(bodyColor: AppColors.chalk, displayColor: AppColors.chalk),
  );
}
