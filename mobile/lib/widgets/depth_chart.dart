import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models.dart';
import '../theme.dart';

/// Depth of the hips (in thigh lengths) over the set. Up = deeper.
/// Shaded band = borderline zone around the depth line; rep windows are tinted
/// by their verdict and numbered at the bottom of each rep.
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
    if (a.timeS.length < 2) return;
    const padL = 6.0, padB = 18.0, padT = 14.0;
    final w = size.width - padL, h = size.height - padB - padT;
    final t0 = a.timeS.first, t1 = a.timeS.last;
    final lo = math.min(a.depth.reduce(math.min), -1.0);
    final hi = math.max(a.depth.reduce(math.max), a.creaseOffset + a.borderlineBand) + 0.1;
    double x(double t) => padL + (t - t0) / (t1 - t0) * w;
    double y(double d) => padT + (hi - d) / (hi - lo) * h;

    // rep windows
    for (final r in a.reps) {
      final c = AppColors.status(r.status).withValues(alpha: 0.10);
      canvas.drawRect(Rect.fromLTRB(x(r.startS), padT, x(r.endS), padT + h), Paint()..color = c);
    }
    // borderline band + depth line
    canvas.drawRect(
      Rect.fromLTRB(padL, y(a.creaseOffset + a.borderlineBand), size.width, y(a.creaseOffset - a.borderlineBand)),
      Paint()..color = AppColors.borderline.withValues(alpha: 0.15),
    );
    canvas.drawLine(Offset(padL, y(a.creaseOffset)), Offset(size.width, y(a.creaseOffset)),
        Paint()
          ..color = AppColors.borderline
          ..strokeWidth = 1);

    // depth curve
    final path = Path()..moveTo(x(a.timeS[0]), y(a.depth[0]));
    for (var i = 1; i < a.timeS.length; i++) {
      path.lineTo(x(a.timeS[i]), y(a.depth[i]));
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
      tp.paint(canvas, Offset(x(a.timeS[i]) - tp.width / 2, y(a.depth[i]) - tp.height - 4));
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
