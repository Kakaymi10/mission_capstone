import 'package:flutter/material.dart';

class ArcaneQuizScreen extends StatefulWidget {
  final String specimenName;
  final String? specimenImageUrl;

  const ArcaneQuizScreen({
    super.key,
    required this.specimenName,
    this.specimenImageUrl,
  });

  @override
  _ArcaneQuizScreenState createState() => _ArcaneQuizScreenState();
}

class _ArcaneQuizScreenState extends State<ArcaneQuizScreen> {
  int _currentQuestionIndex = 0;
  int _score = 0;
  bool _answerSelected = false;
  String? _selectedAnswer;

  // Sample quiz questions - you can replace with your own
  final List<Map<String, dynamic>> _quizQuestions = [
    {
      'question': 'What magical property does this specimen possess?',
      'answers': [
        'Ethereal Glow',
        'Time Distortion',
        'Memory Absorption',
        'Elemental Conjuring',
      ],
      'correctAnswer': 'Ethereal Glow',
    },
    {
      'question': 'Which ancient civilization first documented this specimen?',
      'answers': [
        'Atlanteans',
        'Elders of Mu',
        'Lemurian Archivists',
        'Hyperborean Scholars',
      ],
      'correctAnswer': 'Atlanteans',
    },
    {
      'question': 'How should this specimen be properly stored?',
      'answers': [
        'In a lead-lined box',
        'Under moonlight',
        'Submerged in mercury',
        'Wrapped in silk',
      ],
      'correctAnswer': 'Wrapped in silk',
    },
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Arcane Quiz: ${widget.specimenName}'),
        backgroundColor: Colors.indigo.shade900,
        foregroundColor: Colors.amber,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: RadialGradient(
            colors: [Colors.indigo.shade800, Colors.black],
            center: Alignment.center,
            radius: 1.5,
          ),
        ),
        child: Center(
          child: Container(
            constraints: BoxConstraints(maxWidth: 600),
            padding: EdgeInsets.all(20),
            child: Card(
              color: Colors.indigo.shade900.withOpacity(0.8),
              elevation: 10,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(20),
                side: BorderSide(color: Colors.amber, width: 1),
              ),
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: _buildQuizContent(),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildQuizContent() {
    if (_currentQuestionIndex >= _quizQuestions.length) {
      return Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            'Quiz Complete!',
            style: TextStyle(
              color: Colors.amber,
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
          ),
          SizedBox(height: 20),
          Text(
            'Your arcane knowledge score: $_score/${_quizQuestions.length}',
            style: TextStyle(color: Colors.amber, fontSize: 18),
          ),
          SizedBox(height: 30),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.amber,
              foregroundColor: Colors.indigo.shade900,
            ),
            onPressed: () {
              Navigator.pop(context);
            },
            child: Text('Return to Specimen'),
          ),
        ],
      );
    }

    final currentQuestion = _quizQuestions[_currentQuestionIndex];
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          'Question ${_currentQuestionIndex + 1}/${_quizQuestions.length}',
          style: TextStyle(color: Colors.amber.withOpacity(0.7), fontSize: 16),
        ),
        SizedBox(height: 10),
        Text(
          currentQuestion['question'],
          style: TextStyle(
            color: Colors.amber,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
          textAlign: TextAlign.center,
        ),
        SizedBox(height: 30),
        ...(currentQuestion['answers'] as List<String>).map((answer) {
          bool isCorrect = answer == currentQuestion['correctAnswer'];
          bool isSelected = _selectedAnswer == answer;

          Color buttonColor =
              _answerSelected
                  ? (isCorrect
                      ? Colors.green.withOpacity(0.7)
                      : isSelected
                      ? Colors.red.withOpacity(0.7)
                      : Colors.indigo.shade800)
                  : Colors.indigo.shade800;

          return Padding(
            padding: const EdgeInsets.symmetric(vertical: 8.0),
            child: ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: buttonColor,
                foregroundColor: Colors.white,
                minimumSize: Size(double.infinity, 50),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                  side: BorderSide(
                    color: isSelected ? Colors.amber : Colors.transparent,
                    width: 2,
                  ),
                ),
              ),
              onPressed:
                  _answerSelected
                      ? null
                      : () {
                        setState(() {
                          _selectedAnswer = answer;
                          _answerSelected = true;
                          if (isCorrect) {
                            _score++;
                          }
                        });
                      },
              child: Text(answer),
            ),
          );
        }).toList(),
        SizedBox(height: 20),
        if (_answerSelected)
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.amber,
              foregroundColor: Colors.indigo.shade900,
            ),
            onPressed: () {
              setState(() {
                _currentQuestionIndex++;
                _answerSelected = false;
                _selectedAnswer = null;
              });
            },
            child: Text(
              _currentQuestionIndex < _quizQuestions.length - 1
                  ? 'Next Question'
                  : 'See Results',
            ),
          ),
      ],
    );
  }
}
