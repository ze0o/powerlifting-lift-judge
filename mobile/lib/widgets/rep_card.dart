import 'package:flutter/material.dart';

import '../models.dart';
import '../theme.dart';
import 'ref_lights.dart';

/// One rep: referee lights, verdict, every rule check with its reason,
/// coaching metrics and the neural network's second opinion.
class RepCard extends StatelessWidget {
  final RepVerdict rep;
  final VoidCallback? onSeek; // jump the video to this rep

  const RepCard({super.key, required this.rep, this.onSeek});

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    final m = rep.metrics;
    String fmt(String k, int digits, String unit) =>
        m[k] == null ? '–' : '${m[k]!.toStringAsFixed(digits)}$unit';

    return Card(
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          initiallyExpanded: rep.status != CheckStatus.pass,
          tilePadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 4),
          childrenPadding: const EdgeInsets.fromLTRB(14, 0, 14, 14),
          leading: RefLights(status: rep.status, cards: rep.cards, size: 14),
          title: Text('Rep ${rep.rep}  ·  ${rep.label}',
              style: t.titleMedium?.copyWith(fontWeight: FontWeight.w700, letterSpacing: 0.3)),
          subtitle: Text(
            '${rep.startS.toStringAsFixed(1)}s – ${rep.endS.toStringAsFixed(1)}s',
            style: t.bodySmall?.copyWith(color: AppColors.muted),
          ),
          trailing: onSeek == null
              ? null
              : IconButton(
                  tooltip: 'Watch this rep',
                  icon: const Icon(Icons.play_circle_outline),
                  onPressed: onSeek,
                ),
          children: [
            for (final c in rep.checks)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 4),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(_icon(c.status), size: 18, color: AppColors.status(c.status)),
                    const SizedBox(width: 10),
                    Expanded(child: Text(c.message, style: t.bodyMedium)),
                    if (c.status == CheckStatus.fail)
                      Container(
                        width: 9,
                        height: 13,
                        margin: const EdgeInsets.only(left: 8, top: 3),
                        color: AppColors.card(c.card),
                      ),
                  ],
                ),
              ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: rep.lift == 'deadlift'
                  ? [
                      _Metric('Pull', fmt('pull_s', 2, 's')),
                      _Metric('Lockout knee', fmt('lockout_knee', 0, '°')),
                      _Metric('Lockout hip', fmt('lockout_hip', 0, '°')),
                      _Metric('Bar drift', fmt('bar_drift', 2, ' th')),
                    ]
                  : [
                      _Metric('Descent', fmt('descent_s', 2, 's')),
                      _Metric('Ascent', fmt('ascent_s', 2, 's')),
                      _Metric('Sticking speed', fmt('sticking_speed', 2, ' th/s')),
                      _Metric('Sway', fmt('shoulder_drift', 2, ' th')),
                    ],
            ),
            if (rep.dl != null) ...[
              const SizedBox(height: 12),
              _DlBox(rep: rep),
            ],
          ],
        ),
      ),
    );
  }

  static IconData _icon(CheckStatus s) => switch (s) {
        CheckStatus.pass => Icons.check_circle,
        CheckStatus.fail => Icons.cancel,
        CheckStatus.borderline => Icons.help,
        CheckStatus.unknown => Icons.visibility_off,
      };
}

class _Metric extends StatelessWidget {
  final String label, value;
  const _Metric(this.label, this.value);

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          border: Border.all(color: AppColors.line),
          borderRadius: BorderRadius.circular(4),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(label, style: const TextStyle(color: AppColors.muted, fontSize: 11)),
            Text(value, style: const TextStyle(fontWeight: FontWeight.w600)),
          ],
        ),
      );
}

class _DlBox extends StatelessWidget {
  final RepVerdict rep;
  const _DlBox({required this.rep});

  @override
  Widget build(BuildContext context) {
    final dl = rep.dl!;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: AppColors.stage,
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: rep.disagrees ? AppColors.borderline : AppColors.line),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            const Icon(Icons.psychology_alt_outlined, size: 18, color: AppColors.muted),
            const SizedBox(width: 6),
            Text('Neural network: ${dl.label}', style: const TextStyle(fontWeight: FontWeight.w600)),
          ]),
          if (dl.found.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(dl.found.join(' · '), style: const TextStyle(color: AppColors.muted)),
            ),
          if (rep.disagrees)
            const Padding(
              padding: EdgeInsets.only(top: 6),
              child: Text('Disagrees with the rules: worth a human look.',
                  style: TextStyle(color: AppColors.borderline)),
            ),
        ],
      ),
    );
  }
}
