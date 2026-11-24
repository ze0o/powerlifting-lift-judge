import 'dart:async';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

import '../api.dart';
import '../settings.dart';
import '../theme.dart';
import 'results_screen.dart';

/// Uploads the video, waits for the server, then replaces itself with the results.
class AnalyzingScreen extends StatefulWidget {
  final XFile video;
  final AppSettings settings;

  const AnalyzingScreen({super.key, required this.video, required this.settings});

  @override
  State<AnalyzingScreen> createState() => _AnalyzingScreenState();
}

class _AnalyzingScreenState extends State<AnalyzingScreen> {
  String? _error;

  @override
  void initState() {
    super.initState();
    _run();
  }

  void _fail(String message) {
    if (mounted) setState(() => _error = message);
  }

  Future<void> _run() async {
    // _error starts null, so no setState on the first run (not allowed in initState)
    if (_error != null) setState(() => _error = null);
    final s = widget.settings;
    final api = JudgeApi(s.serverUrl);
    try {
      final result = await api.analyze(widget.video,
          lift: s.lift, modelSize: s.modelSize, creaseOffset: s.creaseOffset);
      if (!mounted) return;
      Navigator.of(context).pushReplacement(MaterialPageRoute(
        builder: (_) => ResultsScreen(
          analysis: result,
          api: api,
          videoUrl: result.videoUrl == null ? null : api.videoUrl(result.videoUrl!),
        ),
      ));
    } on ApiException catch (e) {
      _fail(e.message);
    } on TimeoutException {
      _fail('The server took too long. Try a shorter video or the Nano model.');
    } on http.ClientException {
      // no server, wrong address, or (in a browser) the request was blocked
      _fail("Can't reach the server at ${s.serverUrl}. Is it running? "
          'In Chrome use http://localhost:8000; on a phone use your PC\'s Wi-Fi address.');
    } catch (e) {
      _fail('Something went wrong: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(title: const Text('Judging')),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: _error == null
            ? Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const LinearProgressIndicator(color: AppColors.accent),
                  const SizedBox(height: 24),
                  Text('Analysing your set', style: t.headlineSmall?.copyWith(fontWeight: FontWeight.w700)),
                  const SizedBox(height: 12),
                  for (final step in const [
                    'Uploading the video',
                    'Finding the lifter and tracking 17 body points',
                    'Splitting the set into reps',
                    'Checking the camera angle',
                    'Checking each rep against the IPF rules',
                    'Asking the neural network for a second opinion',
                  ])
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 5),
                      child: Row(children: [
                        const Icon(Icons.chevron_right, size: 18, color: AppColors.muted),
                        const SizedBox(width: 6),
                        Expanded(child: Text(step, style: const TextStyle(color: AppColors.muted))),
                      ]),
                    ),
                  const SizedBox(height: 16),
                  Text(
                    'Model size "${widget.settings.modelSize}". Longer sets and bigger models take longer.',
                    style: t.bodySmall?.copyWith(color: AppColors.muted),
                  ),
                ],
              )
            : Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(Icons.error_outline, color: AppColors.noLift, size: 40),
                  const SizedBox(height: 12),
                  Text(_error!, style: t.titleMedium),
                  const SizedBox(height: 20),
                  Row(children: [
                    FilledButton(onPressed: _run, child: const Text('Try again')),
                    const SizedBox(width: 12),
                    TextButton(onPressed: () => Navigator.pop(context), child: const Text('Back')),
                  ]),
                ],
              ),
      ),
    );
  }
}
