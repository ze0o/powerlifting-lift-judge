import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../models.dart';
import '../theme.dart';
import '../widgets/depth_chart.dart';
import '../widgets/ref_lights.dart';
import '../widgets/rep_card.dart';

class ResultsScreen extends StatefulWidget {
  final Analysis analysis;
  final String? videoUrl; // absolute URL of the annotated video, if any
  final bool isSample;

  const ResultsScreen({super.key, required this.analysis, this.videoUrl, this.isSample = false});

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  VideoPlayerController? _video;

  @override
  void initState() {
    super.initState();
    final url = widget.videoUrl;
    if (url != null) {
      _video = VideoPlayerController.networkUrl(Uri.parse(url))
        ..initialize().then((_) {
          if (mounted) setState(() {});
        })
        ..setLooping(true);
    }
  }

  @override
  void dispose() {
    _video?.dispose();
    super.dispose();
  }

  void _seek(RepVerdict r) {
    final v = _video;
    if (v == null || !v.value.isInitialized) return;
    v.seekTo(Duration(milliseconds: (r.startS * 1000).round()));
    v.play();
  }

  @override
  Widget build(BuildContext context) {
    final a = widget.analysis;
    final t = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(title: Text(widget.isSample ? 'Sample set' : 'Your set')),
      body: a.reps.isEmpty
          ? const _NoReps()
          : ListView(
              padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
              children: [
                _Scoreboard(summary: a.summary),
                const SizedBox(height: 16),
                if (_video != null) ...[
                  _VideoBox(controller: _video!),
                  const SizedBox(height: 16),
                ],
                Text('DEPTH OVER THE SET',
                    style: t.labelMedium?.copyWith(color: AppColors.muted, letterSpacing: 1.2)),
                const SizedBox(height: 6),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: DepthChart(analysis: a),
                  ),
                ),
                const SizedBox(height: 16),
                Text('REPS', style: t.labelMedium?.copyWith(color: AppColors.muted, letterSpacing: 1.2)),
                const SizedBox(height: 6),
                for (final r in a.reps) ...[
                  RepCard(rep: r, onSeek: _video == null ? null : () => _seek(r)),
                  const SizedBox(height: 10),
                ],
                if (!a.dlAvailable)
                  Text('Neural-network second opinion is off on the server (run `liftjudge train-dl`).',
                      style: t.bodySmall?.copyWith(color: AppColors.muted)),
                const SizedBox(height: 8),
                Text('A training aid, not a replacement for referees.',
                    style: t.bodySmall?.copyWith(color: AppColors.muted)),
              ],
            ),
    );
  }
}

class _Scoreboard extends StatelessWidget {
  final Summary summary;
  const _Scoreboard({required this.summary});

  @override
  Widget build(BuildContext context) {
    Widget cell(String label, int n, Color c, CheckStatus s) => Expanded(
          child: Column(
            children: [
              RefLights(status: s, size: 12),
              const SizedBox(height: 8),
              Text('$n', style: TextStyle(fontSize: 30, fontWeight: FontWeight.w800, color: c)),
              Text(label, style: const TextStyle(color: AppColors.muted, fontSize: 12)),
            ],
          ),
        );
    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 16),
        child: Row(children: [
          cell('Good lifts', summary.good, AppColors.good, CheckStatus.pass),
          cell('No lifts', summary.noLift, AppColors.noLift, CheckStatus.fail),
          cell('Borderline', summary.borderline, AppColors.borderline, CheckStatus.borderline),
        ]),
      ),
    );
  }
}

class _VideoBox extends StatelessWidget {
  final VideoPlayerController controller;
  const _VideoBox({required this.controller});

  @override
  Widget build(BuildContext context) {
    if (!controller.value.isInitialized) {
      return const AspectRatio(
        aspectRatio: 16 / 9,
        child: Center(child: CircularProgressIndicator()),
      );
    }
    return ClipRRect(
      borderRadius: BorderRadius.circular(6),
      child: AspectRatio(
        aspectRatio: controller.value.aspectRatio,
        child: Stack(
          alignment: Alignment.bottomCenter,
          children: [
            VideoPlayer(controller),
            VideoProgressIndicator(controller, allowScrubbing: true),
            Positioned.fill(
              child: Material(
                color: Colors.transparent,
                child: InkWell(
                  onTap: () => controller.value.isPlaying ? controller.pause() : controller.play(),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _NoReps extends StatelessWidget {
  const _NoReps();

  @override
  Widget build(BuildContext context) => const Padding(
        padding: EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(Icons.videocam_off_outlined, size: 40, color: AppColors.muted),
            SizedBox(height: 12),
            Text('No reps found', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700)),
            SizedBox(height: 8),
            Text('Film from the side with the camera at hip height, and keep your whole body '
                'and the bar in frame for the whole set.'),
          ],
        ),
      );
}
