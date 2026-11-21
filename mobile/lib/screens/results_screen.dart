import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../api.dart';
import '../models.dart';
import '../theme.dart';
import '../widgets/depth_chart.dart';
import 'guide_screen.dart';
import '../widgets/ref_lights.dart';
import '../widgets/rep_card.dart';

class ResultsScreen extends StatefulWidget {
  final Analysis analysis;
  final JudgeApi? api; // needed to save the upload for training
  final String? videoUrl; // absolute URL of the annotated video, if any
  final bool isSample;

  const ResultsScreen({super.key, required this.analysis, this.api, this.videoUrl, this.isSample = false});

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  VideoPlayerController? _video;
  String? _savedPath;
  bool _saving = false;

  Future<void> _save() async {
    final api = widget.api, job = widget.analysis.jobId;
    if (api == null || job == null) return;
    setState(() => _saving = true);
    try {
      final path = await api.saveForTraining(job);
      if (!mounted) return;
      setState(() => _savedPath = path);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Could not save: $e')));
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

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
    final liftName = a.lift == 'deadlift' ? 'deadlift' : 'squat';
    return Scaffold(
      appBar: AppBar(title: Text(widget.isSample ? 'Sample $liftName set' : 'Your $liftName set')),
      body: a.reps.isEmpty
          ? const _NoReps()
          : ListView(
              padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
              children: [
                if (!a.view.isSide && a.view.message.isNotEmpty) ...[
                  _ViewBanner(view: a.view),
                  const SizedBox(height: 12),
                ],
                _Scoreboard(summary: a.summary),
                const SizedBox(height: 16),
                if (_video != null) ...[
                  _VideoBox(controller: _video!),
                  const SizedBox(height: 16),
                ],
                Text(a.metric == 'bar_height' ? 'BAR HEIGHT OVER THE SET' : 'DEPTH OVER THE SET',
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
                if (a.jobId != null && widget.api != null) ...[
                  const SizedBox(height: 6),
                  _SaveBox(saving: _saving, savedPath: _savedPath, onSave: _save),
                  const SizedBox(height: 12),
                ],
                if (!a.dlAvailable && a.lift == 'squat')
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


class _ViewBanner extends StatelessWidget {
  final CameraView view;
  const _ViewBanner({required this.view});

  @override
  Widget build(BuildContext context) {
    final color = view.kind == 'front' ? AppColors.noLift : AppColors.borderline;
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: color),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Icon(Icons.videocam_outlined, color: color, size: 20),
          const SizedBox(width: 8),
          Expanded(
            child: Text(view.kind == 'front' ? 'Filmed from the front' : 'Filmed at an angle',
                style: TextStyle(color: color, fontWeight: FontWeight.w700)),
          ),
        ]),
        const SizedBox(height: 6),
        Text(view.message),
        Align(
          alignment: Alignment.centerLeft,
          child: TextButton(
            onPressed: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => const GuideScreen())),
            child: const Text('How to film'),
          ),
        ),
      ]),
    );
  }
}

/// Keep this upload on the PC (data/raw/<lift>/) to label it and improve the model.
class _SaveBox extends StatelessWidget {
  final bool saving;
  final String? savedPath;
  final VoidCallback onSave;
  const _SaveBox({required this.saving, required this.savedPath, required this.onSave});

  @override
  Widget build(BuildContext context) {
    if (savedPath != null) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Icon(Icons.check_circle, color: Color(0xFF52B788)),
            const SizedBox(width: 10),
            Expanded(
              child: Text('Saved for training: $savedPath\nNext on the PC: liftjudge label, then check the labels.',
                  style: const TextStyle(height: 1.35)),
            ),
          ]),
        ),
      );
    }
    return OutlinedButton.icon(
      onPressed: saving ? null : onSave,
      icon: saving
          ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
          : const Icon(Icons.save_alt),
      label: const Text('Save for training'),
      style: OutlinedButton.styleFrom(minimumSize: const Size.fromHeight(48)),
    );
  }
}
