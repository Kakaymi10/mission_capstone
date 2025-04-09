// Quiz models
enum QuestionType {
  multipleChoice,
  trueFalse,
  regionIdentification
}

class QuizQuestion {
  final String id;
  final QuestionType type;
  final String text;
  final List<String>? options;
  final String? regionBbox;
  final int correctOption;
  final String? explanation;
  final int pointsValue;

  QuizQuestion({
    required this.id,
    required this.type,
    required this.text,
    this.options,
    this.regionBbox,
    required this.correctOption,
    this.explanation,
    required this.pointsValue,
  });
}