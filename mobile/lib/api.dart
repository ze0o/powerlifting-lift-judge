import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

import 'models.dart';

class ApiException implements Exception {
  final String message;
  ApiException(this.message);
  @override
  String toString() => message;
}

/// Talks to the Python server: `uvicorn liftjudge.api:app --host 0.0.0.0 --port 8000`.
class JudgeApi {
  final String baseUrl; // e.g. http://192.168.1.20:8000
  final http.Client _client;

  JudgeApi(String baseUrl, {http.Client? client})
      : baseUrl = baseUrl.endsWith('/') ? baseUrl.substring(0, baseUrl.length - 1) : baseUrl,
        _client = client ?? http.Client();

  Uri _uri(String path, [Map<String, String>? q]) => Uri.parse('$baseUrl$path').replace(queryParameters: q);

  String videoUrl(String relative) => '$baseUrl$relative';

  /// Returns true if the server answers, and whether its neural network is loaded.
  Future<({bool ok, bool dl})> health() async {
    try {
      final r = await _client.get(_uri('/health')).timeout(const Duration(seconds: 4));
      if (r.statusCode != 200) return (ok: false, dl: false);
      final j = jsonDecode(r.body) as Map<String, dynamic>;
      return (ok: true, dl: j['dl_model'] as bool? ?? false);
    } catch (_) {
      return (ok: false, dl: false);
    }
  }

  Future<Analysis> sample({double? creaseOffset}) async {
    final r = await _client
        .get(_uri('/sample', creaseOffset == null ? null : {'crease_offset': '$creaseOffset'}))
        .timeout(const Duration(seconds: 30));
    return Analysis.fromJson(_decode(r.statusCode, r.body));
  }

  /// Uploads a video and waits for the verdict (pose estimation runs on the server).
  Future<Analysis> analyze(
    XFile video, {
    String lift = 'squat',
    String modelSize = 'n',
    double creaseOffset = 0.07,
  }) async {
    final req = http.MultipartRequest('POST', _uri('/analyze'))
      ..fields['lift'] = lift
      ..fields['model_size'] = modelSize
      ..fields['crease_offset'] = '$creaseOffset'
      // bytes instead of a file path: in a browser there is no file path
      ..files.add(http.MultipartFile.fromBytes('video', await video.readAsBytes(),
          filename: video.name.isEmpty ? 'video.mp4' : video.name));
    final streamed = await _client.send(req).timeout(const Duration(minutes: 10));
    final body = await streamed.stream.bytesToString();
    return Analysis.fromJson(_decode(streamed.statusCode, body));
  }

  Map<String, dynamic> _decode(int status, String body) {
    if (status == 200) return jsonDecode(body) as Map<String, dynamic>;
    String msg = 'Server error ($status)';
    try {
      final detail = (jsonDecode(body) as Map<String, dynamic>)['detail'];
      if (detail is String) msg = detail;
    } catch (_) {}
    throw ApiException(msg);
  }
}
