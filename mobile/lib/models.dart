// Data returned by the Lift Judge API (see src/liftjudge/api.py).

double? _d(Object? v) => v == null ? null : (v as num).toDouble();

enum CheckStatus { pass, fail, borderline, unknown }

CheckStatus _status(String s) => CheckStatus.values.firstWhere(
      (e) => e.name == s,
      orElse: () => CheckStatus.unknown,
    );

/// IPF referee card colours.
enum RefCard { red, blue, yellow }

RefCard _card(String s) => RefCard.values.firstWhere((e) => e.name == s);

class Check {
  final String code;
  final RefCard card;
  final CheckStatus status;
  final String message;
  final double? value;
  final double? threshold;

  const Check({
    required this.code,
    required this.card,
    required this.status,
    required this.message,
    this.value,
    this.threshold,
  });

  factory Check.fromJson(Map<String, dynamic> j) => Check(
        code: j['code'] as String,
        card: _card(j['card'] as String),
        status: _status(j['status'] as String),
        message: j['message'] as String,
        value: _d(j['value']),
        threshold: _d(j['threshold']),
      );
}

/// The neural network's second opinion on a rep.
class DlOpinion {
  final String label;
  final Map<String, double> faults;
  final List<String> found;

  const DlOpinion({required this.label, required this.faults, required this.found});

  bool get noLift => label == 'NO LIFT';

  factory DlOpinion.fromJson(Map<String, dynamic> j) => DlOpinion(
        label: j['label'] as String,
        faults: (j['faults'] as Map<String, dynamic>)
            .map((k, v) => MapEntry(k, (v as num).toDouble())),
        found: (j['found'] as List).cast<String>(),
      );
}

class RepVerdict {
  final int rep; // 1-based for display
  final CheckStatus status;
  final String label;
  final List<RefCard> cards;
  final List<Check> checks;
  final Map<String, double?> metrics;
  final double startS;
  final double bottomS;
  final double endS;
  final DlOpinion? dl;

  const RepVerdict({
    required this.rep,
    required this.status,
    required this.label,
    required this.cards,
    required this.checks,
    required this.metrics,
    required this.startS,
    required this.bottomS,
    required this.endS,
    this.dl,
  });

  /// True when the network and the rules give a different call (a human should look).
  bool get disagrees =>
      dl != null && status != CheckStatus.borderline && dl!.noLift != (status == CheckStatus.fail);

  factory RepVerdict.fromJson(Map<String, dynamic> j) => RepVerdict(
        rep: (j['rep'] as num).toInt() + 1,
        status: _status(j['status'] as String),
        label: j['label'] as String,
        cards: (j['cards'] as List).map((c) => _card(c as String)).toList(),
        checks: (j['checks'] as List)
            .map((c) => Check.fromJson(c as Map<String, dynamic>))
            .toList(),
        metrics: (j['metrics'] as Map<String, dynamic>).map((k, v) => MapEntry(k, _d(v))),
        startS: _d(j['start_s']) ?? 0,
        bottomS: _d(j['bottom_s']) ?? 0,
        endS: _d(j['end_s']) ?? 0,
        dl: j['dl'] == null ? null : DlOpinion.fromJson(j['dl'] as Map<String, dynamic>),
      );
}

class Summary {
  final int reps, good, noLift, borderline;
  const Summary(this.reps, this.good, this.noLift, this.borderline);

  factory Summary.fromJson(Map<String, dynamic> j) => Summary(
        (j['reps'] as num).toInt(),
        (j['good'] as num).toInt(),
        (j['no_lift'] as num).toInt(),
        (j['borderline'] as num).toInt(),
      );
}

class Analysis {
  final String lift;
  final Summary summary;
  final List<RepVerdict> reps;
  final List<double> timeS;
  final List<double> depth;
  final double creaseOffset;
  final double borderlineBand;
  final String? videoUrl; // relative to the server, e.g. /videos/abc.mp4
  final bool dlAvailable;

  const Analysis({
    required this.lift,
    required this.summary,
    required this.reps,
    required this.timeS,
    required this.depth,
    required this.creaseOffset,
    required this.borderlineBand,
    required this.videoUrl,
    required this.dlAvailable,
  });

  factory Analysis.fromJson(Map<String, dynamic> j) {
    final series = j['series'] as Map<String, dynamic>;
    final cfg = j['config'] as Map<String, dynamic>;
    List<double> nums(String k) =>
        (series[k] as List).map((v) => (v as num?)?.toDouble() ?? 0).toList();
    return Analysis(
      lift: j['lift'] as String,
      summary: Summary.fromJson(j['summary'] as Map<String, dynamic>),
      reps: (j['reps'] as List)
          .map((r) => RepVerdict.fromJson(r as Map<String, dynamic>))
          .toList(),
      timeS: nums('time_s'),
      depth: nums('depth'),
      creaseOffset: _d(cfg['crease_offset']) ?? 0.07,
      borderlineBand: _d(cfg['borderline_band']) ?? 0.04,
      videoUrl: j['video_url'] as String?,
      dlAvailable: j['dl_available'] as bool? ?? false,
    );
  }
}
