import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// App settings, saved on the phone.
class AppSettings extends ChangeNotifier {
  // In Chrome (Flutter web) the server is on the same PC: localhost.
  // The Android emulator reaches the PC at 10.0.2.2. On a real phone use the
  // PC's Wi-Fi address, e.g. http://192.168.1.20:8000 (run `ipconfig` on Windows).
  static const defaultServer = kIsWeb ? 'http://localhost:8000' : 'http://10.0.2.2:8000';

  String serverUrl = defaultServer;
  String modelSize = 'n'; // n, s, m, l: bigger = more precise, slower
  double creaseOffset = 0.07;

  Future<void> load() async {
    final p = await SharedPreferences.getInstance();
    serverUrl = p.getString('serverUrl') ?? defaultServer;
    modelSize = p.getString('modelSize') ?? 'n';
    creaseOffset = p.getDouble('creaseOffset') ?? 0.07;
    notifyListeners();
  }

  Future<void> save({String? serverUrl, String? modelSize, double? creaseOffset}) async {
    final p = await SharedPreferences.getInstance();
    if (serverUrl != null) {
      this.serverUrl = serverUrl.trim();
      await p.setString('serverUrl', this.serverUrl);
    }
    if (modelSize != null) {
      this.modelSize = modelSize;
      await p.setString('modelSize', modelSize);
    }
    if (creaseOffset != null) {
      this.creaseOffset = creaseOffset;
      await p.setDouble('creaseOffset', creaseOffset);
    }
    notifyListeners();
  }
}
