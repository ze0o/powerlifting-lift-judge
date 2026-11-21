import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:lift_judge/api.dart';
import 'package:lift_judge/models.dart';
import 'package:lift_judge/screens/guide_screen.dart';
import 'package:lift_judge/screens/results_screen.dart';
import 'package:lift_judge/theme.dart';
import 'package:lift_judge/widgets/ref_lights.dart';

/// assets/sample*.json are real responses from the Python API (`GET /sample?lift=...`).
Map<String, dynamic> sampleJson([String file = 'assets/sample.json']) =>
    jsonDecode(File(file).readAsStringSync()) as Map<String, dynamic>;

void main() {
  group('models', () {
    test('parse the API response', () {
      final a = Analysis.fromJson(sampleJson());
      expect(a.summary.reps, 3);
      expect(a.summary.good, 1);
      expect(a.summary.noLift, 2);
      expect(a.reps.map((r) => r.label), ['GOOD LIFT', 'NO LIFT', 'NO LIFT']);
      expect(a.reps[0].rep, 1); // API is 0-based, app shows 1-based
      expect(a.reps[1].cards, [RefCard.red]);
      expect(a.reps[2].cards, [RefCard.blue]);
      expect(a.reps[0].checks.length, 6);
      expect(a.metric, 'depth');
      expect(a.timeS.length, a.values.length);
      expect(a.creaseOffset, 0.07);
      expect(a.view.isSide, isTrue);
      expect(a.reps[0].startS < a.reps[0].bottomS, isTrue);
    });

    test('parse a deadlift response', () {
      final a = Analysis.fromJson(sampleJson('assets/sample_deadlift.json'));
      expect(a.lift, 'deadlift');
      expect(a.metric, 'bar_height');
      expect(a.creaseOffset, isNull); // no depth line for deadlifts
      expect(a.reps.map((r) => r.label), ['GOOD LIFT', 'NO LIFT', 'NO LIFT']);
      expect(a.reps[1].cards, [RefCard.red]); // soft lockout
      expect(a.reps[2].cards, [RefCard.blue]); // hitch
      expect(a.reps[0].metrics['lockout_knee'], greaterThan(170));
      expect(a.reps[0].dl, isNull);
    });

    test('network opinion and disagreement', () {
      final a = Analysis.fromJson(sampleJson());
      expect(a.reps[2].dl!.found, contains('double bounce in the hole'));
      expect(a.reps.any((r) => r.disagrees), isFalse);

      final j = sampleJson();
      (j['reps'] as List)[0]['dl']['label'] = 'NO LIFT'; // rules say good, network says no
      expect(Analysis.fromJson(j).reps[0].disagrees, isTrue);
    });

    test('null metrics (not measurable) are allowed', () {
      final j = sampleJson();
      (j['reps'] as List)[0]['metrics']['sticking_speed'] = null;
      expect(Analysis.fromJson(j).reps[0].metrics['sticking_speed'], isNull);
    });
  });

  group('api', () {
    test('sample request and error messages', () async {
      final client = MockClient((req) async {
        if (req.url.path == '/sample') return http.Response(jsonEncode(sampleJson()), 200);
        return http.Response(jsonEncode({'detail': "Couldn't see the lifter's legs."}), 422);
      });
      final api = JudgeApi('http://pc:8000/', client: client);
      expect((await api.sample()).summary.reps, 3);
      await expectLater(api.saveForTraining('abc'), throwsA(isA<ApiException>()));
      expect(api.videoUrl('/videos/abc.mp4'), 'http://pc:8000/videos/abc.mp4');
      final health = await api.health();
      expect(health.ok, isFalse);
    });
  });

  group('widgets', () {
    testWidgets('results screen shows scoreboard, chart and every rep', (tester) async {
      tester.view.physicalSize = const Size(1080, 4000);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      final a = Analysis.fromJson(sampleJson());
      await tester.pumpWidget(MaterialApp(theme: buildTheme(), home: ResultsScreen(analysis: a, isSample: true)));
      await tester.pumpAndSettle();
      expect(find.text('Good lifts'), findsOneWidget);
      expect(find.text('Rep 1  ·  GOOD LIFT'), findsOneWidget);
      expect(find.text('Rep 2  ·  NO LIFT'), findsOneWidget);
      // no-lift reps open by default and show the reason
      expect(find.textContaining('High: hip crease did not go below'), findsOneWidget);
      expect(find.byType(RefLights), findsWidgets);
    });

    testWidgets('front view shows a warning; uploads can be saved for training', (tester) async {
      tester.view.physicalSize = const Size(1080, 5000);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      final j = sampleJson()
        ..['view'] = {'kind': 'front', 'ratio': 0.7, 'message': 'Front/back view: film from the side.'}
        ..['job_id'] = 'abc123';
      var saved = false;
      final api = JudgeApi('http://pc:8000', client: MockClient((req) async {
        saved = req.url.path == '/jobs/abc123/save';
        return http.Response(jsonEncode({'saved': true, 'path': 'data/raw/squat/x.mp4'}), 200);
      }));
      await tester.pumpWidget(MaterialApp(theme: buildTheme(), home: ResultsScreen(analysis: Analysis.fromJson(j), api: api)));
      await tester.pumpAndSettle();
      expect(find.text('Filmed from the front'), findsOneWidget);
      await tester.tap(find.text('Save for training'));
      await tester.pumpAndSettle();
      expect(saved, isTrue);
      expect(find.textContaining('data/raw/squat/x.mp4'), findsOneWidget);
    });

    testWidgets('guide screen', (tester) async {
      await tester.pumpWidget(MaterialApp(theme: buildTheme(), home: const GuideScreen()));
      expect(find.text('How to film'), findsOneWidget);
      expect(find.text('At hip height'), findsOneWidget);
    });

    testWidgets('no reps shows filming advice', (tester) async {
      final j = sampleJson()..['reps'] = [];
      await tester.pumpWidget(MaterialApp(home: ResultsScreen(analysis: Analysis.fromJson(j))));
      expect(find.text('No reps found'), findsOneWidget);
    });
  });
}
