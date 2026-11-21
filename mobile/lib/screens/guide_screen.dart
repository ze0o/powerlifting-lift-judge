import 'package:flutter/material.dart';

import '../theme.dart';

/// How to film so the judge can see what a referee sees.
class GuideScreen extends StatelessWidget {
  const GuideScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    final label = t.labelMedium?.copyWith(color: AppColors.muted, letterSpacing: 1.2);
    return Scaffold(
      appBar: AppBar(title: const Text('How to film')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
        children: [
          Text('Put the camera where a side referee sits',
              style: t.headlineSmall?.copyWith(fontWeight: FontWeight.w800, height: 1.15)),
          const SizedBox(height: 8),
          Text(
            'Depth and lockout are about where your hip is compared with your knee. Only a side view '
            'shows that. From the front, the thighs point at the camera and the hip and knee overlap.',
            style: t.bodyMedium?.copyWith(color: AppColors.muted),
          ),
          const SizedBox(height: 16),
          const Card(
            child: Padding(
              padding: EdgeInsets.all(12),
              child: AspectRatio(aspectRatio: 1.25, child: CustomPaint(painter: _TopDownPainter())),
            ),
          ),
          const SizedBox(height: 10),
          const _Legend(),
          const SizedBox(height: 22),
          Text('CAMERA', style: label),
          const SizedBox(height: 6),
          const _Rule(Icons.height, 'At hip height', 'About 80–100 cm off the floor. Higher or lower makes the '
              'hip look higher or lower than it is.'),
          const _Rule(Icons.straighten, '3–5 m away', 'Close enough that you fill most of the frame, far enough '
              'that feet, head and the whole bar stay in it for the full set.'),
          const _Rule(Icons.crop_rotate, 'Square to the lifter', 'The camera looks straight at your side, at 90° to the '
              'line between your feet. The app warns you if it detects an angled or front view.'),
          const _Rule(Icons.videocam_outlined, 'Steady, landscape, 30 or 60 fps', 'Phone on a tripod or a fixed stand. '
              'Start recording before you unrack and stop after you rack or set the bar down.'),
          const SizedBox(height: 18),
          Text('IN FRAME', style: label),
          const SizedBox(height: 6),
          const _Rule(Icons.accessibility_new, 'Your whole body', 'Head to feet, every rep. If the feet leave the frame '
              'the judge can\'t check for steps.'),
          const _Rule(Icons.groups_2_outlined, 'Spotters behind you, not in front', 'They stand between the camera and '
              'your knees at the bottom of a squat, which is exactly when the judge needs to see them.'),
          const _Rule(Icons.checkroom, 'Clothes that show the hip and knee', 'A singlet or fitted shorts. Baggy shorts '
              'and a long shirt hide the hip crease; a belt is fine.'),
          const _Rule(Icons.light_mode_outlined, 'Good light, plain background', 'Light on you, not behind you. Fewer people '
              'walking behind you means fewer tracking mistakes.'),
          const SizedBox(height: 18),
          Text('PER LIFT', style: label),
          const SizedBox(height: 6),
          const _Rule(Icons.airline_seat_legroom_extra, 'Squat', 'Side view at hip height. Film the side whose hip and knee '
              'are not hidden by the bar plates.'),
          const _Rule(Icons.fitness_center, 'Deadlift', 'Side view, camera slightly behind the bar so the plates don\'t '
              'hide your knees and hips at lockout.'),
          const _Rule(Icons.weekend_outlined, 'Bench press', 'Coming later: it needs bar tracking, because the plates hide '
              'the wrists from the side.'),
          const SizedBox(height: 18),
          Text('SETTINGS', style: label),
          const SizedBox(height: 6),
          const _Rule(Icons.tune, 'Close depth calls: use Small or Medium', 'The Nano pose model is fast but its hip and '
              'knee points move a little from frame to frame. Bigger models are steadier.'),
        ],
      ),
    );
  }
}

class _Rule extends StatelessWidget {
  final IconData icon;
  final String title, body;
  const _Rule(this.icon, this.title, this.body);

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 7),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Icon(icon, size: 22, color: AppColors.chalk),
          const SizedBox(width: 12),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(title, style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15)),
              const SizedBox(height: 2),
              Text(body, style: const TextStyle(color: AppColors.muted, height: 1.35)),
            ]),
          ),
        ]),
      );
}

class _Legend extends StatelessWidget {
  const _Legend();

  @override
  Widget build(BuildContext context) {
    Widget item(Color c, String t) => Row(mainAxisSize: MainAxisSize.min, children: [
          Container(width: 10, height: 10, decoration: BoxDecoration(color: c, shape: BoxShape.circle)),
          const SizedBox(width: 6),
          Text(t, style: const TextStyle(fontSize: 12, color: AppColors.muted)),
        ]);
    return Wrap(spacing: 16, runSpacing: 6, children: [
      item(const Color(0xFF52B788), 'Side: best'),
      item(AppColors.borderline, '45°: less precise'),
      item(AppColors.noLift, 'Front / back: depth not judged'),
    ]);
  }
}

/// Seen from above: the lifter and bar in the middle, camera positions around.
class _TopDownPainter extends CustomPainter {
  const _TopDownPainter();

  @override
  void paint(Canvas canvas, Size size) {
    final c = Offset(size.width / 2, size.height / 2);
    final r = size.shortestSide * 0.36;
    final ring = Paint()
      ..color = AppColors.line
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    canvas.drawCircle(c, r, ring);

    // bar (left-right) and lifter (seen from above: shoulders + head), facing up = towards the front camera
    canvas.drawLine(c.translate(-r * 0.55, 0), c.translate(r * 0.55, 0),
        Paint()
          ..color = AppColors.muted
          ..strokeWidth = 4
          ..strokeCap = StrokeCap.round);
    for (final dx in [-r * 0.55, r * 0.55]) {
      canvas.drawRRect(
          RRect.fromRectAndRadius(Rect.fromCenter(center: c.translate(dx, 0), width: 8, height: 26),
              const Radius.circular(2)),
          Paint()..color = AppColors.muted);
    }
    canvas.drawOval(Rect.fromCenter(center: c, width: r * 0.62, height: r * 0.26), Paint()..color = AppColors.chalk);
    canvas.drawCircle(c.translate(0, -r * 0.02), r * 0.1, Paint()..color = AppColors.stage);
    _text(canvas, 'lifter', c.translate(0, r * 0.28), AppColors.muted, 11);

    // cameras: front (top), 45° (top-right), side (right), back (bottom)
    const good = Color(0xFF52B788);
    _camera(canvas, c.translate(r * 1.18, 0), 180, good, 'SIDE ✓', r);
    _camera(canvas, c.translate(-r * 1.18, 0), 0, good, 'SIDE ✓', r);
    _camera(canvas, c.translate(r * 0.83, -r * 0.83), 135, AppColors.borderline, '45°', r);
    _camera(canvas, c.translate(0, -r * 1.18), 90, AppColors.noLift, 'FRONT ✗', r);

    // dashed line of sight for the side camera
    final sight = Paint()
      ..color = good.withValues(alpha: 0.6)
      ..strokeWidth = 1.5;
    for (double x = c.dx + r * 0.35; x < c.dx + r * 1.02; x += 10) {
      canvas.drawLine(Offset(x, c.dy), Offset(x + 5, c.dy), sight);
    }
  }

  void _camera(Canvas canvas, Offset p, double facingDeg, Color color, String label, double r) {
    canvas.save();
    canvas.translate(p.dx, p.dy);
    canvas.rotate(facingDeg * 3.1415926 / 180);
    final body = Paint()..color = color;
    canvas.drawRRect(RRect.fromRectAndRadius(Rect.fromCenter(center: Offset.zero, width: 22, height: 16),
        const Radius.circular(3)), body);
    final lens = Path()
      ..moveTo(11, -5)
      ..lineTo(19, -9)
      ..lineTo(19, 9)
      ..lineTo(11, 5)
      ..close();
    canvas.drawPath(lens, body);
    canvas.restore();
    // the front camera sits at the top: label above it, the others below
    _text(canvas, label, p.translate(0, facingDeg == 90 ? -22.0 : 22.0), color, 12, bold: true);
  }

  void _text(Canvas canvas, String s, Offset centre, Color color, double size, {bool bold = false}) {
    final tp = TextPainter(
      text: TextSpan(
          text: s, style: TextStyle(color: color, fontSize: size, fontWeight: bold ? FontWeight.w700 : FontWeight.w400)),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, centre - Offset(tp.width / 2, tp.height / 2));
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
