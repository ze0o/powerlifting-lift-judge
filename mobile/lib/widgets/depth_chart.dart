import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models.dart';
import '../theme.dart';

/// Squat: depth of the hips (thigh lengths) over the set, up = deeper, with the
/// depth line and its borderline band. Deadlift: bar height above the feet.
/// Rep windows are tinted by their verdict and numbered at the key moment of
/// each rep (bottom of the squat, top of the pull).
class DepthChart extends StatelessWidget {
  final Analysis analysis;
  final double height;

  const DepthChart({super.key, required this.analysis, this.height = 170});

  @override
  Widget build(BuildContext context) => SizedBox(
        height: height,
        width: double.infinity,
        child: CustomPaint(painter: _DepthPainter(analysis)),
      );
}

class _DepthPainter extends CustomPainter {
  final Analysis a;
  _DepthPainter(this.a);

  @override
  void paint(Canvas canvas, Size size) {
    if (a.timeS.length < 2 || a.values.length != a.timeS.length) return;
    const padL = 6.0, padB = 18.0, padT = 14.0;
    final w = size.width - padL, h = size.height - padB - padT;
    final t0 = a.timeS.first, t1 = a.timeS.last;
    final off = a.creaseOffset, band = a.borderlineBand ?? 0;
    final vmin = a.values.reduce(math.min), vmax = a.values.reduce(math.max);
    final lo = off == null ? vmin - 0.1 : math.min(vmin, -1.0);
    final hi = (off == null ? vmax : math.max(vmax, off + band)) + 0.1;
    double x(double t) => padL + (t - t0) / (t1 - t0) * w;
    double y(double d) => padT + (hi - d) / (hi - lo) * h;

    // rep windows
    for (final r in a.reps) {
      final c = AppColors.status(r.status).withValues(alpha: 0.10);
      canvas.drawRect(Rect.fromLTRB(x(r.startS), padT, x(r.endS), padT + h), Paint()..color = c);
    }
    // squat: borderline band + depth line (not drawn for a front view: depth isn't judged there)
    if (off != null && a.view.kind != 'front') {
      canvas.drawRect(
        Rect.fromLTRB(padL, y(off + band), size.width, y(off - band)),
        Paint()..color = AppColors.borderline.withValues(alpha: 0.15),
      );
      canvas.drawLine(Offset(padL, y(off)), Offset(size.width, y(off)),
          Paint()
            ..color = AppColors.borderline
            ..strokeWidth = 1);
    }

    // curve
    final path = Path()..moveTo(x(a.timeS[0]), y(a.values[0]));
    for (var i = 1; i < a.timeS.length; i++) {
      path.lineTo(x(a.timeS[i]), y(a.values[i]));
    }
    canvas.drawPath(
        path,
        Paint()
          ..color = AppColors.chalk
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2
          ..strokeJoin = StrokeJoin.round);

    // rep numbers at the bottom of each rep
    for (final r in a.reps) {
      final i = _nearest(r.bottomS);
      final tp = TextPainter(
        text: TextSpan(
            text: '${r.rep}',
            style: TextStyle(color: AppColors.status(r.status), fontSize: 12, fontWeight: FontWeight.w700)),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(x(a.timeS[i]) - tp.width / 2, y(a.values[i]) - tp.height - 4));
    }

    // time axis
    final axis = TextPainter(
      text: TextSpan(
          text: '0s', style: const TextStyle(color: AppColors.muted, fontSize: 10)),
      textDirection: TextDirection.ltr,
    )..layout();
    axis.paint(canvas, Offset(padL, size.height - axis.height));
    final end = TextPainter(
      text: TextSpan(
          text: '${t1.toStringAsFixed(1)}s', style: const TextStyle(color: AppColors.muted, fontSize: 10)),
      textDirection: TextDirection.ltr,
    )..layout();
    end.paint(canvas, Offset(size.width - end.width, size.height - end.height));
  }

  int _nearest(double t) {
    var best = 0;
    for (var i = 1; i < a.timeS.length; i++) {
      if ((a.timeS[i] - t).abs() < (a.timeS[best] - t).abs()) best = i;
    }
    return best;
  }

  @override
  bool shouldRepaint(covariant _DepthPainter old) => old.a != a;
}
