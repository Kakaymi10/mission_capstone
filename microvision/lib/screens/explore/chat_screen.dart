// ChatScreen for AI-powered analysis of specimens
import 'dart:convert';
import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:flutter/rendering.dart';
import 'models.dart';

class ChatScreen extends StatefulWidget {
  final Map<String, dynamic> specimen;
  final Rect? selectedRegion;
  final Future<Uint8List?> Function() captureCanvas;

  const ChatScreen({
    super.key,
    required this.specimen,
    this.selectedRegion,
    required this.captureCanvas,
  });

  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  bool _isLoadingChat = false;
  TextEditingController _chatController = TextEditingController();
  List<Map<String, String>> _chatHistory = [];

  // Canvas display variables
  ui.Image? _specimenImage;
  bool _isImageLoaded = false;
  late Size _imageSize;
  Uint8List? _canvasImageBytes;

  // For displaying the image with selected region
  final GlobalKey _canvasKey = GlobalKey();

  // Suggested prompts
  final List<String> _suggestedPrompts = [
    "What is this structure and what is its function?",
    "What cell type is this?",
    "What are the key characteristics of this structure?",
    "Is this a normal or abnormal structure?",
    "What staining technique was used here?",
  ];

  @override
  void initState() {
    super.initState();
    // If we have a selected region, add a message about it
    if (widget.selectedRegion != null) {
      _chatHistory.add({
        'role': 'system',
        'content': 'Region selected. Ask a question about this specific area.',
      });
    }

    // Load the specimen image
    _loadImage();

    // Capture the canvas with annotations
    _captureCanvasImage();
  }

  @override
  void dispose() {
    _chatController.dispose();
    super.dispose();
  }

  // Load the specimen image
  Future<void> _loadImage() async {
    if (widget.specimen['image_url'] == null) {
      setState(() => _isImageLoaded = false);
      return;
    }

    final ImageProvider provider = NetworkImage(widget.specimen['image_url']);
    final ImageStream stream = provider.resolve(const ImageConfiguration());

    stream.addListener(
      ImageStreamListener((ImageInfo info, bool _) {
        setState(() {
          _imageSize = Size(
            info.image.width.toDouble(),
            info.image.height.toDouble(),
          );
          _specimenImage = info.image;
          _isImageLoaded = true;
        });
      }),
    );
  }

  // Capture the canvas image from the previous screen
  Future<void> _captureCanvasImage() async {
    try {
      final bytes = await widget.captureCanvas();
      if (bytes != null) {
        setState(() {
          _canvasImageBytes = bytes;
        });
      }
    } catch (e) {
      print('Error capturing canvas image: $e');
    }
  }

  // Chat with the LLaVA model about a specific region
  Future<void> _chatWithLLaVA({String? question}) async {
    // Get the question from the controller if not provided
    final String userQuestion = question ?? _chatController.text.trim();
    if (userQuestion.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please enter a question'),
          behavior: SnackBarBehavior.floating,
        ),
      );
      return;
    }

    // Add the user message to chat history
    setState(() {
      _chatHistory.add({'role': 'user', 'content': userQuestion});
      _isLoadingChat = true;
      // Clear the text field if this is a direct input (not suggested prompt)
      if (question == null) {
        _chatController.clear();
      }
    });

    try {
      // Use the already captured canvas image if available
      Uint8List? imageBytes = _canvasImageBytes;

      // If not available, try to capture it now
      if (imageBytes == null) {
        imageBytes = await widget.captureCanvas();
        if (imageBytes == null) {
          throw Exception('Failed to capture canvas for chat');
        }

        // Save it for future use
        setState(() {
          _canvasImageBytes = imageBytes;
        });
      }

      // Prepare bounding box string if a region is selected
      String? bboxString;
      if (widget.selectedRegion != null) {
        bboxString =
            '${widget.selectedRegion!.left.round()},${widget.selectedRegion!.top.round()},${widget.selectedRegion!.right.round()},${widget.selectedRegion!.bottom.round()}';
      }

      // Create multipart request for the chat_region endpoint
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('http://127.0.0.1:8000/chat_region/'),
      );

      // Add the image and question
      request.files.add(
        http.MultipartFile.fromBytes(
          'image',
          imageBytes,
          filename: 'canvas_image.png',
        ),
      );
      request.fields['question'] = userQuestion;

      // Add bbox if available
      if (bboxString != null) {
        request.fields['bbox'] = bboxString;
      }

      // Optional label
      if (widget.selectedRegion != null) {
        request.fields['label'] = 'Selected Region';
      }

      // Send the request
      final response = await request.send();
      final responseString = await response.stream.bytesToString();

      if (response.statusCode != 200) {
        throw Exception(
          'Failed to get chat response: ${response.statusCode} - $responseString',
        );
      }

      // Parse the response
      final Map<String, dynamic> responseData;
      try {
        responseData = jsonDecode(responseString);
      } catch (e) {
        throw Exception('Invalid JSON response: $e');
      }

      if (!responseData.containsKey('answer')) {
        throw Exception('Invalid response format: missing answer field');
      }

      // Add the assistant response to chat history
      setState(() {
        _chatHistory.add({
          'role': 'assistant',
          'content': responseData['answer'],
        });
        _isLoadingChat = false;
      });
    } catch (e) {
      print('Chat error: $e');
      // Add error message to chat history
      setState(() {
        _chatHistory.add({
          'role': 'assistant',
          'content': 'Error: ${e.toString()}',
        });
        _isLoadingChat = false;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error getting response: ${e.toString()}'),
          behavior: SnackBarBehavior.floating,
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  // Custom painter for displaying the image with selected region
  Widget _buildImageDisplay() {
    if (_canvasImageBytes != null) {
      // If we have the captured canvas image with annotations, show that
      return Container(
        height: 200,
        width: double.infinity,
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey.shade300),
        ),
        child: Image.memory(_canvasImageBytes!, fit: BoxFit.contain),
      );
    } else if (_isImageLoaded && _specimenImage != null) {
      // Otherwise show the original specimen image with the region overlay
      return Container(
        height: 200,
        width: double.infinity,
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey.shade300),
        ),
        child: CustomPaint(
          painter: ChatImagePainter(
            image: _specimenImage!,
            selectedRegion: widget.selectedRegion,
          ),
          size: Size(double.infinity, 200),
        ),
      );
    } else {
      // Loading or error state
      return Container(
        height: 200,
        width: double.infinity,
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey.shade300),
          color: Colors.grey.shade100,
        ),
        child: const Center(child: CircularProgressIndicator()),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AI Chat Analysis'),
        backgroundColor: Theme.of(context).primaryColor,
        foregroundColor: Colors.white,
        elevation: 4,
      ),
      body: Column(
        children: [
          // Specimen info banner
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            color: Theme.of(context).primaryColor.withOpacity(0.1),
            child: Row(
              children: [
                const Icon(Icons.science, size: 24),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        widget.specimen['name'] ?? 'Unnamed Specimen',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      if (widget.specimen['collection'] != null)
                        Text(
                          widget.specimen['collection'],
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                    ],
                  ),
                ),
                if (widget.selectedRegion != null)
                  Chip(
                    avatar: const Icon(Icons.crop_free, size: 16),
                    label: const Text('Region Selected'),
                    backgroundColor: Colors.green.withOpacity(0.2),
                  ),
              ],
            ),
          ),

          // Specimen image with selected region
          _buildImageDisplay(),

          // Divider
          const Divider(height: 1),

          // Suggested prompts
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Suggested Questions:',
                  style: Theme.of(
                    context,
                  ).textTheme.bodySmall?.copyWith(fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children:
                      _suggestedPrompts.map((prompt) {
                        return InkWell(
                          onTap: () => _chatWithLLaVA(question: prompt),
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 6,
                            ),
                            decoration: BoxDecoration(
                              color: Theme.of(
                                context,
                              ).primaryColor.withOpacity(0.1),
                              borderRadius: BorderRadius.circular(16),
                            ),
                            child: Text(
                              prompt,
                              style: Theme.of(context).textTheme.bodySmall,
                            ),
                          ),
                        );
                      }).toList(),
                ),
              ],
            ),
          ),

          // Chat history
          Expanded(
            child:
                _chatHistory.isEmpty
                    ? Center(
                      child: Text(
                        'Ask a question about the specimen',
                        style: Theme.of(
                          context,
                        ).textTheme.bodyMedium?.copyWith(color: Colors.grey),
                      ),
                    )
                    : ListView.builder(
                      padding: const EdgeInsets.all(16),
                      itemCount: _chatHistory.length + (_isLoadingChat ? 1 : 0),
                      itemBuilder: (context, index) {
                        if (index == _chatHistory.length && _isLoadingChat) {
                          return const Center(
                            child: Padding(
                              padding: EdgeInsets.all(16.0),
                              child: CircularProgressIndicator(),
                            ),
                          );
                        }

                        final message = _chatHistory[index];
                        final isUser = message['role'] == 'user';
                        final isSystem = message['role'] == 'system';

                        if (isSystem) {
                          return Container(
                            margin: const EdgeInsets.symmetric(vertical: 8.0),
                            padding: const EdgeInsets.all(8.0),
                            decoration: BoxDecoration(
                              color: Colors.amber.withOpacity(0.1),
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(
                                color: Colors.amber.withOpacity(0.5),
                              ),
                            ),
                            child: Text(
                              message['content'] ?? '',
                              style: Theme.of(context).textTheme.bodySmall
                                  ?.copyWith(fontStyle: FontStyle.italic),
                              textAlign: TextAlign.center,
                            ),
                          );
                        }

                        return Padding(
                          padding: const EdgeInsets.symmetric(vertical: 8.0),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              CircleAvatar(
                                backgroundColor:
                                    isUser ? Colors.blue : Colors.green,
                                radius: 16,
                                child: Icon(
                                  isUser ? Icons.person : Icons.smart_toy,
                                  color: Colors.white,
                                  size: 16,
                                ),
                              ),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Container(
                                  padding: const EdgeInsets.all(12),
                                  decoration: BoxDecoration(
                                    color:
                                        isUser
                                            ? Colors.blue.withOpacity(0.1)
                                            : Colors.green.withOpacity(0.1),
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                  child: Text(
                                    message['content'] ?? '',
                                    style:
                                        Theme.of(context).textTheme.bodyMedium,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
          ),

          // Chat input
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.grey[100],
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.05),
                  offset: const Offset(0, -1),
                  blurRadius: 3,
                ),
              ],
            ),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _chatController,
                    decoration: const InputDecoration(
                      hintText: 'Ask a question...',
                      border: OutlineInputBorder(),
                      contentPadding: EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 12,
                      ),
                    ),
                    onSubmitted: (_) => _chatWithLLaVA(),
                    enabled: !_isLoadingChat,
                    maxLines: 2,
                    minLines: 1,
                  ),
                ),
                const SizedBox(width: 8),
                IconButton(
                  icon: const Icon(Icons.send),
                  onPressed: _isLoadingChat ? null : _chatWithLLaVA,
                  color: Theme.of(context).primaryColor,
                  style: IconButton.styleFrom(
                    backgroundColor: Theme.of(
                      context,
                    ).primaryColor.withOpacity(0.1),
                    padding: const EdgeInsets.all(12),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// Custom painter for displaying the image with selected region
class ChatImagePainter extends CustomPainter {
  final ui.Image image;
  final Rect? selectedRegion;

  ChatImagePainter({required this.image, this.selectedRegion});

  @override
  void paint(Canvas canvas, Size size) {
    // Calculate the aspect ratio to fit the image properly
    final double imageAspectRatio = image.width / image.height;
    final double canvasAspectRatio = size.width / size.height;

    double drawWidth;
    double drawHeight;
    double x = 0;
    double y = 0;

    if (imageAspectRatio > canvasAspectRatio) {
      // Image is wider than canvas
      drawWidth = size.width;
      drawHeight = size.width / imageAspectRatio;
      y = (size.height - drawHeight) / 2;
    } else {
      // Image is taller than canvas
      drawHeight = size.height;
      drawWidth = size.height * imageAspectRatio;
      x = (size.width - drawWidth) / 2;
    }

    // Draw the image
    final src = Rect.fromLTWH(
      0,
      0,
      image.width.toDouble(),
      image.height.toDouble(),
    );
    final dst = Rect.fromLTWH(x, y, drawWidth, drawHeight);
    canvas.drawImageRect(image, src, dst, Paint());

    // Draw the selected region if available
    if (selectedRegion != null) {
      // Scale the region coordinates to match the displayed image size
      final double scaleX = drawWidth / image.width;
      final double scaleY = drawHeight / image.height;

      final scaledRegion = Rect.fromLTRB(
        x + selectedRegion!.left * scaleX,
        y + selectedRegion!.top * scaleY,
        x + selectedRegion!.right * scaleX,
        y + selectedRegion!.bottom * scaleY,
      );

      // Draw a semi-transparent fill
      final paint =
          Paint()
            ..color = Colors.green.withOpacity(0.3)
            ..style = PaintingStyle.fill;
      canvas.drawRect(scaledRegion, paint);

      // Draw a border
      final borderPaint =
          Paint()
            ..color = Colors.green
            ..style = PaintingStyle.stroke
            ..strokeWidth = 2.0;
      canvas.drawRect(scaledRegion, borderPaint);
    }
  }

  @override
  bool shouldRepaint(covariant ChatImagePainter oldDelegate) {
    return oldDelegate.image != image ||
        oldDelegate.selectedRegion != selectedRegion;
  }
}
