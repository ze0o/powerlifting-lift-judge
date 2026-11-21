import 'dart:convert';

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image_picker/image_picker.dart';

import '../api.dart';
import '../models.dart';
import '../settings.dart';
import '../theme.dart';
import '../widgets/ref_lights.dart';
import 'analyzing_screen.dart';
import 'guide_screen.dart';
import 'results_screen.dart';
import 'settings_screen.dart';

class HomeScreen extends StatefulWidget {
  final AppSettings settings;
  const HomeScreen({super.key, required this.settings});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _picker = ImagePicker();
  bool? _online;
  bool _dl = false;

  @override
  void initState() {
    super.initState();
    widget.settings.addListener(_ping);
    // setState isn't allowed inside initState: check the server after the first frame.
    WidgetsBinding.instance.addPostFrameCallback((_) => _ping());
  }

  @override
  void dispose() {
    widget.settings.removeListener(_ping);
    super.dispose();
  }

  Future<void> _ping() async {
    if (!mounted) return;
    setState(() => _online = null);
    final h = await JudgeApi(widget.settings.serverUrl).health();
    if (!mounted) return;
    setState(() {
      _online = h.ok;
      _dl = h.dl;
    });
  }

  Future<void> _pick(ImageSource source) async {
    final file = await _picker.pickVideo(source: source, maxDuration: const Duration(minutes: 2));
    if (file == null || !mounted) return;
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => AnalyzingScreen(video: file, settings: widget.settings),
    ));
  }

  /// Sample works offline: a result bundled with the app (synthetic 3-rep set).
  Future<void> _sample() async {
    final asset = widget.settings.lift == 'deadlift' ? 'assets/sample_deadlift.json' : 'assets/sample.json';
    final json = jsonDecode(await rootBundle.loadString(asset)) as Map<String, dynamic>;
    if (!mounted) return;
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => ResultsScreen(analysis: Analysis.fromJson(json), isSample: true),
    ));
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Lift Judge', style: TextStyle(fontWeight: FontWeight.w800, letterSpacing: 0.5)),
        actions: [
          IconButton(
            tooltip: 'Settings',
            icon: const Icon(Icons.tune),
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute(builder: (_) => SettingsScreen(settings: widget.settings)),
            ),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 32),
        children: [
          const Align(alignment: Alignment.centerLeft, child: RefLights(status: CheckStatus.pass, size: 26)),
          const SizedBox(height: 18),
          Text('Was that a good lift?',
              style: t.headlineMedium?.copyWith(fontWeight: FontWeight.w800, height: 1.1)),
          const SizedBox(height: 8),
          Text(
            'Film your lift from the side. Every rep is checked against the IPF rules, with the '
            'reason for every call.',
            style: t.bodyLarge?.copyWith(color: AppColors.muted),
          ),
          const SizedBox(height: 18),
          SegmentedButton<String>(
            segments: const [
              ButtonSegment(value: 'squat', label: Text('Squat')),
              ButtonSegment(value: 'deadlift', label: Text('Deadlift')),
              ButtonSegment(value: 'bench', label: Text('Bench · soon'), enabled: false),
            ],
            selected: {widget.settings.lift},
            onSelectionChanged: (s) => widget.settings.save(lift: s.first),
          ),
          const SizedBox(height: 6),
          Text(
            widget.settings.lift == 'deadlift'
                ? 'Checks: knees locked, standing erect, downward movement, hitching, feet.'
                : 'Checks: depth, knees locked, double bounce, downward movement, feet.',
            style: t.bodySmall?.copyWith(color: AppColors.muted),
          ),
          const SizedBox(height: 16),
          _ServerChip(online: _online, dl: _dl, url: widget.settings.serverUrl, onRetry: _ping),
          const SizedBox(height: 20),
          if (!kIsWeb) ...[
            _BigButton(
              icon: Icons.videocam,
              title: 'Record a set',
              subtitle: 'Opens the camera',
              primary: true,
              onTap: () => _pick(ImageSource.camera),
            ),
            const SizedBox(height: 10),
          ],
          _BigButton(
            icon: Icons.video_library_outlined,
            title: 'Choose a video',
            subtitle: kIsWeb ? 'From your computer (.mp4, .mov)' : 'From your gallery',
            primary: kIsWeb,
            onTap: () => _pick(ImageSource.gallery),
          ),
          const SizedBox(height: 10),
          _BigButton(
            icon: Icons.science_outlined,
            title: 'See a sample result',
            subtitle: 'Works offline',
            onTap: _sample,
          ),
          const SizedBox(height: 24),
          Text('FILMING TIPS', style: t.labelMedium?.copyWith(color: AppColors.muted, letterSpacing: 1.2)),
          const SizedBox(height: 8),
          Card(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(14, 14, 14, 4),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const _Tip('Side view, camera at hip height, not from the front'),
                  const _Tip('Camera square to the lifter, not angled'),
                  const _Tip('Whole body and the bar in frame for the whole set'),
                  const _Tip('Spotters behind the lifter, phone on a tripod'),
                  TextButton.icon(
                    onPressed: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => const GuideScreen())),
                    icon: const Icon(Icons.menu_book_outlined, size: 18),
                    label: const Text('Full filming guide'),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ServerChip extends StatelessWidget {
  final bool? online;
  final bool dl;
  final String url;
  final VoidCallback onRetry;
  const _ServerChip({required this.online, required this.dl, required this.url, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    final (color, text) = switch (online) {
      null => (AppColors.muted, 'Checking server...'),
      true => (const Color(0xFF52B788), dl ? 'Server connected · neural network on' : 'Server connected'),
      false => (AppColors.noLift, 'Server offline: $url'),
    };
    return InkWell(
      onTap: onRetry,
      child: Row(children: [
        Icon(Icons.circle, size: 10, color: color),
        const SizedBox(width: 8),
        Expanded(child: Text(text, style: TextStyle(color: color, fontSize: 13))),
        if (online == false) const Text('Retry', style: TextStyle(fontSize: 13)),
      ]),
    );
  }
}

class _BigButton extends StatelessWidget {
  final IconData icon;
  final String title, subtitle;
  final bool primary;
  final VoidCallback onTap;
  const _BigButton({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
    this.primary = false,
  });

  @override
  Widget build(BuildContext context) => Material(
        color: primary ? AppColors.accent : AppColors.panel,
        borderRadius: BorderRadius.circular(6),
        child: InkWell(
          borderRadius: BorderRadius.circular(6),
          onTap: onTap,
          child: Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(6),
              border: primary ? null : Border.all(color: AppColors.line),
            ),
            child: Row(children: [
              Icon(icon, size: 26),
              const SizedBox(width: 14),
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(title, style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w700)),
                  Text(subtitle,
                      style: TextStyle(color: primary ? Colors.white70 : AppColors.muted, fontSize: 13)),
                ]),
              ),
              const Icon(Icons.chevron_right),
            ]),
          ),
        ),
      );
}

class _Tip extends StatelessWidget {
  final String text;
  const _Tip(this.text);

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(children: [
          const Icon(Icons.check, size: 16, color: AppColors.muted),
          const SizedBox(width: 8),
          Expanded(child: Text(text)),
        ]),
      );
}
