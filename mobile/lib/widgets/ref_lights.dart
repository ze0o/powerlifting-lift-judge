import 'package:flutter/material.dart';

import '../models.dart';
import '../theme.dart';

/// Three referee lights like on a meet platform: white = good, red = no lift.
/// Borderline shows amber in the middle (the judge refuses to call it).
/// Under a red light, the failure card colours are shown.
class RefLights extends StatelessWidget {
  final CheckStatus status;
  final List<RefCard> cards;
  final double size;

  const RefLights({super.key, required this.status, this.cards = const [], this.size = 22});

  @override
  Widget build(BuildContext context) {
    final lights = switch (status) {
      CheckStatus.pass => [AppColors.good, AppColors.good, AppColors.good],
      CheckStatus.fail => [AppColors.noLift, AppColors.noLift, AppColors.noLift],
      _ => [AppColors.good, AppColors.borderline, AppColors.noLift],
    };
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            for (final c in lights)
              Container(
                width: size,
                height: size,
                margin: EdgeInsets.symmetric(horizontal: size * 0.12),
                decoration: BoxDecoration(
                  color: c,
                  shape: BoxShape.circle,
                  boxShadow: [BoxShadow(color: c.withValues(alpha: 0.45), blurRadius: size * 0.5)],
                ),
              ),
          ],
        ),
        if (cards.isNotEmpty) ...[
          SizedBox(height: size * 0.3),
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              for (final c in cards)
                Container(
                  width: size * 0.55,
                  height: size * 0.8,
                  margin: EdgeInsets.symmetric(horizontal: size * 0.1),
                  decoration: BoxDecoration(
                    color: AppColors.card(c),
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
            ],
          ),
        ],
      ],
    );
  }
}
