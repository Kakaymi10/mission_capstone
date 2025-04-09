import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart' show decodeImageFromList;
import 'package:http/http.dart' as http;
import 'package:microvision/screens/explore/quizz_models.dart';
import 'models.dart'; // Ensure your models.dart file is updated to include quiz models

class QuizService {
  static const String _baseUrl = 'http://127.0.0.1:8000';

  // Generate quiz questions based on specimen image
  Future<List<QuizQuestion>> generateQuiz(String imageUrl) async {
    try {
      // Download the image first
      final imageResponse = await http.get(Uri.parse(imageUrl));
      if (imageResponse.statusCode != 200) {
        throw Exception(
          'Failed to download image: ${imageResponse.statusCode}',
        );
      }

      // Get segmentation data to identify structures
      final segmentationData = await _getSegmentation(imageResponse.bodyBytes);

      // Generate questions using LLaVA model
      return await _generateQuestionsWithLLaVA(
        imageResponse.bodyBytes,
        segmentationData,
      );
    } catch (e) {
      print('Error generating quiz: $e');
      // Return fallback questions if generation fails
      return _getFallbackQuestions();
    }
  }

  // Get segmentation data for the image using real FastSAM model
  Future<List<Map<String, dynamic>>> _getSegmentation(
    Uint8List imageBytes,
  ) async {
    try {
      // Analyze the image to automatically detect interesting regions
      // We'll create bounding boxes for significant structures
      final image = await decodeImageFromList(imageBytes);

      // Create several bounding boxes at different positions/scales to identify various structures
      // In a production app, you'd use image analysis to identify these regions dynamically
      final imgWidth = image.width.toDouble();
      final imgHeight = image.height.toDouble();

      // Define regions of interest based on image dimensions
      final List<String> bboxesList = [
        '${imgWidth * 0.1},${imgHeight * 0.1},${imgWidth * 0.4},${imgHeight * 0.4}',
        '${imgWidth * 0.5},${imgHeight * 0.2},${imgWidth * 0.8},${imgHeight * 0.5}',
        '${imgWidth * 0.2},${imgHeight * 0.6},${imgWidth * 0.5},${imgHeight * 0.9}',
        '${imgWidth * 0.3},${imgHeight * 0.3},${imgWidth * 0.7},${imgHeight * 0.7}',
      ];

      // Create multipart request to connect to your FastAPI endpoint
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$_baseUrl/segment/'),
      );

      // Add image file
      request.files.add(
        http.MultipartFile.fromBytes(
          'image',
          imageBytes,
          filename: 'specimen.png',
        ),
      );

      // Add bounding boxes parameter - using our defined regions
      request.fields['bboxes'] = bboxesList.join(';');

      // Send request and get response
      final response = await request.send();
      final responseBytes = await response.stream.toBytes();

      if (response.statusCode != 200) {
        throw Exception('Failed to get segmentation: ${response.statusCode}');
      }

      // Process the segmentation results from FastSAM
      // In a production app, you'd analyze the segmentation mask to extract precise regions

      // For now, return the bounding boxes we sent
      List<Map<String, dynamic>> regions = [];
      for (int i = 0; i < bboxesList.length; i++) {
        regions.add({'label': 'Structure ${i + 1}', 'bbox': bboxesList[i]});
      }

      return regions;
    } catch (e) {
      print('Error getting segmentation: $e');
      return [];
    }
  }

  // Generate questions using LLaVA model with real data from your API
  Future<List<QuizQuestion>> _generateQuestionsWithLLaVA(
    Uint8List imageBytes,
    List<Map<String, dynamic>> regions,
  ) async {
    List<QuizQuestion> questions = [];

    try {
      // First, let's get a general description using your /describe/ endpoint
      final descriptionRequest = http.MultipartRequest(
        'POST',
        Uri.parse('$_baseUrl/describe/'),
      );

      descriptionRequest.files.add(
        http.MultipartFile.fromBytes(
          'image',
          imageBytes,
          filename: 'specimen.png',
        ),
      );

      // Include anatomical annotations if available in the specimen data
      if (regions.isNotEmpty) {
        final annotations = regions.map((r) => r['label']).join(', ');
        descriptionRequest.fields['annotations'] = annotations;
      }

      final descriptionResponse = await descriptionRequest.send();
      final descriptionData = await descriptionResponse.stream.bytesToString();

      if (descriptionResponse.statusCode != 200) {
        throw Exception(
          'Failed to get description: ${descriptionResponse.statusCode}',
        );
      }

      final descriptionJson = jsonDecode(descriptionData);
      final description = descriptionJson['description'] as String;

      print(
        'LLaVA description: ${description.substring(0, min(100, description.length))}...',
      );

      // Use the LLaVA model to generate plausible quiz questions directly
      final questionGenerationRequest = http.MultipartRequest(
        'POST',
        Uri.parse('$_baseUrl/chat_region/'),
      );

      questionGenerationRequest.files.add(
        http.MultipartFile.fromBytes(
          'image',
          imageBytes,
          filename: 'specimen_questions.png',
        ),
      );

      // Ask LLaVA to generate question options
      questionGenerationRequest.fields['question'] =
          'Generate a multiple choice question about what type of specimen is shown in this image. ' +
          'Provide 4 options with the correct answer first, labeled A, B, C, D.';
      questionGenerationRequest.fields['bbox'] = '0,0,1000,1000'; // Full image
      questionGenerationRequest.fields['label'] = 'Full Specimen';

      final questionResponse = await questionGenerationRequest.send();
      final questionData = await questionResponse.stream.bytesToString();

      if (questionResponse.statusCode == 200) {
        final questionJson = jsonDecode(questionData);
        final llavaResponse = questionJson['answer'] as String;

        // Extract question and options using text processing
        final questionInfo = _extractQuestionAndOptions(llavaResponse);

        if (questionInfo != null) {
          questions.add(
            QuizQuestion(
              id: '1',
              type: QuestionType.multipleChoice,
              text:
                  questionInfo['question'] ??
                  'What type of specimen is shown in this image?',
              options:
                  questionInfo['options'] ??
                  _generateBasicOptions('Tissue sample'),
              correctOption:
                  0, // First option (A) is correct based on our prompt
              explanation: description.split('.').take(2).join('.') + '.',
              pointsValue: 10,
            ),
          );
        } else {
          // Fallback if parsing fails
          questions.add(
            QuizQuestion(
              id: '1',
              type: QuestionType.multipleChoice,
              text: 'What type of specimen is shown in this image?',
              options: _generateBasicOptions(
                description.split(' ').take(3).join(' '),
              ),
              correctOption: 0,
              explanation: description,
              pointsValue: 10,
            ),
          );
        }
      } else {
        // Fallback if API call fails
        questions.add(
          QuizQuestion(
            id: '1',
            type: QuestionType.multipleChoice,
            text: 'What type of specimen is shown in this image?',
            options: _generateBasicOptions('Tissue sample'),
            correctOption: 0,
            explanation: description,
            pointsValue: 10,
          ),
        );
      }

      // Create structure identification questions for each region using your chat_region endpoint
      int questionId = 2;
      for (var region in regions) {
        try {
          // Query the chat_region endpoint to get information about this structure
          final regionRequest = http.MultipartRequest(
            'POST',
            Uri.parse('$_baseUrl/chat_region/'),
          );

          regionRequest.files.add(
            http.MultipartFile.fromBytes(
              'image',
              imageBytes,
              filename: 'specimen_region.png',
            ),
          );

          regionRequest.fields['question'] =
              'What is this specific structure highlighted in the green box and what is its function?';
          regionRequest.fields['bbox'] = region['bbox'];
          regionRequest.fields['label'] = region['label'];

          final regionResponse = await regionRequest.send();
          final regionData = await regionResponse.stream.bytesToString();

          if (regionResponse.statusCode != 200) {
            print('Region API error: ${regionResponse.statusCode}');
            continue; // Skip this region if request fails
          }

          final regionJson = jsonDecode(regionData);
          final regionAnswer = regionJson['answer'] as String;

          // Now ask LLaVA to generate a multiple choice question about this region
          final regionQuestionRequest = http.MultipartRequest(
            'POST',
            Uri.parse('$_baseUrl/chat_region/'),
          );

          regionQuestionRequest.files.add(
            http.MultipartFile.fromBytes(
              'image',
              imageBytes,
              filename: 'specimen_region_question.png',
            ),
          );

          regionQuestionRequest.fields['question'] =
              'Generate a multiple choice question identifying the structure highlighted in the green box. ' +
              'Provide 4 options with the correct answer first, labeled A, B, C, D.';
          regionQuestionRequest.fields['bbox'] = region['bbox'];
          regionQuestionRequest.fields['label'] = region['label'];

          final regionQuestionResponse = await regionQuestionRequest.send();

          if (regionQuestionResponse.statusCode == 200) {
            final regionQuestionData =
                await regionQuestionResponse.stream.bytesToString();
            final regionQuestionJson = jsonDecode(regionQuestionData);
            final regionQuestionText = regionQuestionJson['answer'] as String;

            // Extract question and options
            final regionQuestionInfo = _extractQuestionAndOptions(
              regionQuestionText,
            );

            if (regionQuestionInfo != null) {
              // Create a question about this region
              questions.add(
                QuizQuestion(
                  id: questionId.toString(),
                  type: QuestionType.regionIdentification,
                  text:
                      regionQuestionInfo['question'] ??
                      'Identify the structure highlighted in this region:',
                  regionBbox: region['bbox'],
                  options:
                      regionQuestionInfo['options'] ??
                      _generateBasicOptions('Cell structure'),
                  correctOption: 0, // First option is correct
                  explanation: regionAnswer,
                  pointsValue: 15,
                ),
              );
            }
          }

          // Create a function question for the same region
          final functionQuestionRequest = http.MultipartRequest(
            'POST',
            Uri.parse('$_baseUrl/chat_region/'),
          );

          functionQuestionRequest.files.add(
            http.MultipartFile.fromBytes(
              'image',
              imageBytes,
              filename: 'specimen_function_question.png',
            ),
          );

          functionQuestionRequest.fields['question'] =
              'Generate a multiple choice question about the function of the structure highlighted in the green box. ' +
              'Provide 4 options with the correct answer first, labeled A, B, C, D.';
          functionQuestionRequest.fields['bbox'] = region['bbox'];
          functionQuestionRequest.fields['label'] = region['label'];

          final functionResponse = await functionQuestionRequest.send();

          if (functionResponse.statusCode == 200) {
            final functionData = await functionResponse.stream.bytesToString();
            final functionJson = jsonDecode(functionData);
            final functionText = functionJson['answer'] as String;

            // Extract question and options
            final functionInfo = _extractQuestionAndOptions(functionText);

            if (functionInfo != null) {
              questions.add(
                QuizQuestion(
                  id: (questionId + 1).toString(),
                  type: QuestionType.multipleChoice,
                  text:
                      functionInfo['question'] ??
                      'What is the function of this structure?',
                  options:
                      functionInfo['options'] ??
                      _generateBasicOptions('Cellular transport'),
                  correctOption: 0, // First option is correct
                  explanation: _extractFunctionFromText(regionAnswer),
                  pointsValue: 10,
                ),
              );
            }
          }

          questionId += 2;
        } catch (e) {
          print('Error processing region $questionId: $e');
          continue;
        }
      }

      // Generate a true/false question using LLaVA
      final tfQuestionRequest = http.MultipartRequest(
        'POST',
        Uri.parse('$_baseUrl/chat_region/'),
      );

      tfQuestionRequest.files.add(
        http.MultipartFile.fromBytes(
          'image',
          imageBytes,
          filename: 'specimen_tf_question.png',
        ),
      );

      tfQuestionRequest.fields['question'] =
          'Generate a true/false question about this specimen with the answer being TRUE.';
      tfQuestionRequest.fields['bbox'] = '0,0,1000,1000'; // Full image

      final tfResponse = await tfQuestionRequest.send();

      if (tfResponse.statusCode == 200) {
        final tfData = await tfResponse.stream.bytesToString();
        final tfJson = jsonDecode(tfData);
        final tfText = tfJson['answer'] as String;

        // Extract just the question part
        String tfQuestion = tfText;
        if (tfText.contains('?')) {
          tfQuestion = tfText.split('?')[0] + '?';
        } else if (tfText.contains('.')) {
          tfQuestion = tfText.split('.')[0] + '.';
        }

        questions.add(
          QuizQuestion(
            id: questionId.toString(),
            type: QuestionType.trueFalse,
            text: tfQuestion,
            correctOption: 0, // True is correct (based on our prompt)
            explanation:
                'Based on the specimen characteristics visible in the image.',
            pointsValue: 5,
          ),
        );
      }

      return questions;
    } catch (e) {
      print('Error generating questions with LLaVA: $e');
      return _getFallbackQuestions();
    }
  }

  // Extract question and options from LLaVA response
  Map<String, dynamic>? _extractQuestionAndOptions(String text) {
    try {
      // Look for a question mark
      int questionEndIndex = text.indexOf('?');
      if (questionEndIndex == -1) return null;

      String question = text.substring(0, questionEndIndex + 1).trim();

      // Extract options - look for A., B., C., D. patterns
      List<String> options = [];
      RegExp optionRegex = RegExp(
        r'([A-D])\.\s*(.*?)(?=\s*[A-D]\.|$)',
        dotAll: true,
      );

      final matches = optionRegex.allMatches(text);
      for (var match in matches) {
        if (match.group(2) != null) {
          options.add(match.group(2)!.trim());
        }
      }

      // If we couldn't extract options, return null
      if (options.length < 2) return null;

      return {'question': question, 'options': options};
    } catch (e) {
      print('Error extracting question and options: $e');
      return null;
    }
  }

  // Extract the function description from text
  String _extractFunctionFromText(String text) {
    // Look for sentences containing function-related keywords
    final sentences = text.split('.');

    for (var sentence in sentences) {
      if (sentence.toLowerCase().contains('function') ||
          sentence.toLowerCase().contains('role') ||
          sentence.toLowerCase().contains('responsible for') ||
          sentence.toLowerCase().contains('purpose')) {
        return sentence.trim() + '.';
      }
    }

    // Return first two sentences if no specific function found
    if (sentences.length >= 2) {
      return sentences.take(2).join('.') + '.';
    }

    return text;
  }

  // Helper methods for generating question content

  // Helper for when text extraction fails
  List<String> _generateBasicOptions(String correctAnswer) {
    final List<String> commonStructures = [
      'Epithelial tissue',
      'Connective tissue',
      'Muscle tissue',
      'Nervous tissue',
      'Nucleus',
      'Cell membrane',
      'Mitochondria',
      'Golgi apparatus',
      'Endoplasmic reticulum',
      'Lysosome',
      'Blood vessel',
      'Neuron',
      'Fibroblast',
      'Red blood cell',
      'White blood cell',
      'Bone tissue',
      'Cartilage',
    ];

    // Generate a list with the correct answer and 3 random other options
    List<String> options = [correctAnswer];

    // Add unique random options
    final random = Random();
    while (options.length < 4) {
      final randomOption =
          commonStructures[random.nextInt(commonStructures.length)];
      if (!options.contains(randomOption)) {
        options.add(randomOption);
      }
    }

    return options;
  }

  // Helper function for string length
  int min(int a, int b) {
    return a < b ? a : b;
  }

  // Fallback questions if API calls fail
  List<QuizQuestion> _getFallbackQuestions() {
    return [
      QuizQuestion(
        id: '1',
        type: QuestionType.multipleChoice,
        text:
            'What type of cellular structure is primarily visible in this specimen?',
        options: [
          'Epithelial tissue',
          'Connective tissue',
          'Muscle tissue',
          'Nervous tissue',
        ],
        correctOption: 0,
        explanation: 'The specimen shows characteristics of epithelial tissue.',
        pointsValue: 10,
      ),
      QuizQuestion(
        id: '2',
        type: QuestionType.trueFalse,
        text:
            'This specimen is stained using H&E (Hematoxylin and Eosin) staining technique.',
        correctOption: 0, // True
        explanation:
            'H&E staining is one of the most common staining methods in histology.',
        pointsValue: 5,
      ),
      QuizQuestion(
        id: '3',
        type: QuestionType.multipleChoice,
        text: 'Which structure is responsible for protein synthesis in cells?',
        options: ['Ribosomes', 'Mitochondria', 'Nucleus', 'Golgi apparatus'],
        correctOption: 0,
        explanation:
            'Ribosomes are the cellular structures where protein synthesis occurs.',
        pointsValue: 10,
      ),
    ];
  }

  // Submit quiz results to backend (optional)
  Future<bool> submitQuizResults(
    String userId,
    String specimenId,
    int score,
    int maxScore,
  ) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/submit_quiz_results/'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'user_id': userId,
          'specimen_id': specimenId,
          'score': score,
          'max_score': maxScore,
          'timestamp': DateTime.now().toIso8601String(),
        }),
      );

      return response.statusCode == 200;
    } catch (e) {
      print('Error submitting quiz results: $e');
      return false;
    }
  }
}
