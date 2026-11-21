import 'package:flutter/material.dart';

import '../api.dart';
import '../settings.dart';
import '../theme.dart';

class SettingsScreen extends StatefulWidget {
  final AppSettings settings;
  const SettingsScreen({super.key, required this.settings});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _url = TextEditingController(text: widget.settings.serverUrl);
  late String _size = widget.settings.modelSize;
  late double _offset = widget.settings.creaseOffset;
  String? _test;

  @override
  void dispose() {
    _url.dispose();
    super.dispose();
  }

  Future<void> _testConnection() async {
    setState(() => _test = 'Testing...');
    final h = await JudgeApi(_url.text.trim()).health();
    if (!mounted) return;
    setState(() => _test = h.ok
        ? 'Connected${h.dl ? ' · neural network loaded' : ' · neural network not loaded'}'
        : 'No answer. Check the address and that the server is running.');
  }

  Future<void> _save() async {
    await widget.settings.save(serverUrl: _url.text, modelSize: _size, creaseOffset: _offset);
    if (mounted) Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    final label = t.labelMedium?.copyWith(color: AppColors.muted, letterSpacing: 1.2);
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text('SERVER', style: label),
          const SizedBox(height: 8),
          TextField(
            controller: _url,
            keyboardType: TextInputType.url,
            decoration: const InputDecoration(
              border: OutlineInputBorder(),
              hintText: 'http://192.168.1.20:8000',
              helperText: "Chrome: http://localhost:8000 · Emulator: http://10.0.2.2:8000 · "
                  "Phone: your PC's Wi-Fi address (ipconfig)",
            ),
          ),
          const SizedBox(height: 8),
          Row(children: [
            OutlinedButton(onPressed: _testConnection, child: const Text('Test connection')),
            const SizedBox(width: 12),
            if (_test != null) Expanded(child: Text(_test!, style: t.bodySmall)),
          ]),
          const SizedBox(height: 28),
          Text('POSE MODEL', style: label),
          const SizedBox(height: 8),
          SegmentedButton<String>(
            segments: const [
              ButtonSegment(value: 'n', label: Text('Nano')),
              ButtonSegment(value: 's', label: Text('Small')),
              ButtonSegment(value: 'm', label: Text('Medium')),
              ButtonSegment(value: 'l', label: Text('Large')),
            ],
            selected: {_size},
            onSelectionChanged: (s) => setState(() => _size = s.first),
          ),
          const SizedBox(height: 6),
          Text('Bigger = more precise hip and knee points for close depth calls, but slower.',
              style: t.bodySmall?.copyWith(color: AppColors.muted)),
          const SizedBox(height: 28),
          Text('DEPTH LINE  ·  ${_offset.toStringAsFixed(2)} thighs', style: label),
          Slider(
            value: _offset,
            min: 0,
            max: 0.2,
            divisions: 20,
            onChanged: (v) => setState(() => _offset = v),
          ),
          Text(
            'How far below the knee joint the hip joint has to go for the hip crease to pass the top '
            'of the knee. Default 0.07; calibrate it on your labelled videos.',
            style: t.bodySmall?.copyWith(color: AppColors.muted),
          ),
          const SizedBox(height: 32),
          FilledButton(onPressed: _save, child: const Text('Save')),
        ],
      ),
    );
  }
}
