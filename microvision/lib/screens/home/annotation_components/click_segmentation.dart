import 'dart:math' as math;
import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'dart:io';

// Model class for segmentation data
class SegmentationMask {
  final String id;
  final Offset clickPoint; // Original click point
  final bool isForeground; // Whether it's a foreground or background point
  String note; // Annotation text
  Color color; // Color for this specific mask
  Uint8List maskImage; // The actual segmentation mask image

  SegmentationMask({
    required this.id,
    required this.clickPoint,
    required this.maskImage,
    this.isForeground = true,
    this.note = '',
    Color? color,
  }) : color = color ?? _getRandomColor();

  // Generate a random color for the mask
  static Color _getRandomColor() {
    final random = math.Random();
    return Color.fromRGBO(
      random.nextInt(200) + 55, // Red component (55-255)
      random.nextInt(200) + 55, // Green component (55-255)
      random.nextInt(200) + 55, // Blue component (55-255)
      0.5, // Alpha (transparency)
    );
  }

  // Convert to map for serialization
  Map<String, dynamic> toJson() {
    // We'll use base64 encoding to store the image data in JSON
    String base64MaskImage = base64Encode(maskImage);

    return {
      'id': id,
      'clickX': clickPoint.dx,
      'clickY': clickPoint.dy,
      'isForeground': isForeground,
      'note': note,
      'maskImageBase64': base64MaskImage,
      'color': {
        'r': color.red,
        'g': color.green,
        'b': color.blue,
        'a': color.alpha,
      },
    };
  }

  // Factory method to create from JSON
  factory SegmentationMask.fromJson(Map<String, dynamic> json) {
    // Decode base64 mask image
    final Uint8List maskImage = base64Decode(json['maskImageBase64']);

    // Create color from color components
    final colorData = json['color'];
    final Color color = Color.fromRGBO(
      colorData['r'],
      colorData['g'],
      colorData['b'],
      colorData['a'].toDouble(),
    );

    return SegmentationMask(
      id: json['id'],
      clickPoint: Offset(json['clickX'].toDouble(), json['clickY'].toDouble()),
      maskImage: maskImage,
      isForeground: json['isForeground'],
      note: json['note'],
      color: color,
    );
  }
}

// Custom painter for displaying segmentation masks
class SegmentationDisplayPainter extends CustomPainter {
  final List<SegmentationMask> masks;
  final String? selectedMaskId;
  final double scale;
  final Uint8List? currentProcessingMask; // Current processing mask image

  SegmentationDisplayPainter({
    required this.masks,
    this.selectedMaskId,
    this.scale = 1.0,
    this.currentProcessingMask,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // First, if we have a current processing mask, draw it
    if (currentProcessingMask != null) {
      _drawMaskImage(
        canvas,
        size,
        currentProcessingMask!,
        Colors.blue.withOpacity(0.5),
      );
    }

    // Then draw all saved masks
    for (final mask in masks) {
      final bool isSelected = mask.id == selectedMaskId;
      final maskColor =
          isSelected
              ? Colors.green.withOpacity(0.5)
              : mask.color.withOpacity(0.5);

      // Draw the mask image
      _drawMaskImage(canvas, size, mask.maskImage, maskColor);

      // Draw click point
      final pointPaint =
          Paint()
            ..color = isSelected ? Colors.green : mask.color
            ..strokeWidth = 2.0 / scale
            ..style = PaintingStyle.fill;

      canvas.drawCircle(mask.clickPoint, 4.0 / scale, pointPaint);

      // Draw foreground/background indicator
      final iconPaint =
          Paint()
            ..color = mask.isForeground ? Colors.white : Colors.black
            ..strokeWidth = 1.0 / scale
            ..style = PaintingStyle.fill;

      canvas.drawCircle(mask.clickPoint, 2.0 / scale, iconPaint);

      // Draw note label if not empty
      if (mask.note.isNotEmpty) {
        _drawNoteLabel(canvas, mask.clickPoint, mask.note, scale, isSelected);
      }
    }
  }

  // Helper method to draw a mask image
  void _drawMaskImage(
    Canvas canvas,
    Size size,
    Uint8List maskBytes,
    Color color,
  ) {
    // We'll use a FutureBuilder in the widget to convert bytes to image
    // Here we assume the conversion is done elsewhere and we're drawing directly
    ui.decodeImageFromList(maskBytes, (ui.Image image) {
      final Paint paint =
          Paint()..colorFilter = ColorFilter.mode(color, BlendMode.srcATop);
      canvas.drawImage(image, Offset.zero, paint);
    });
  }

  void _drawNoteLabel(
    Canvas canvas,
    Offset point,
    String note,
    double scale,
    bool isSelected,
  ) {
    final textSpan = TextSpan(
      text: note,
      style: TextStyle(
        color: Colors.white,
        fontSize: 14.0 / scale,
        fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
      ),
    );

    final textPainter = TextPainter(
      text: textSpan,
      textDirection: TextDirection.ltr,
    );

    textPainter.layout();

    // Background for text
    final textBackgroundRect = Rect.fromLTWH(
      point.dx,
      point.dy - (textPainter.height + 4.0 / scale),
      textPainter.width + 8.0 / scale,
      textPainter.height + 4.0 / scale,
    );

    final backgroundPaint =
        Paint()
          ..color = Colors.black.withOpacity(0.7)
          ..style = PaintingStyle.fill;

    canvas.drawRect(textBackgroundRect, backgroundPaint);

    // Draw text
    textPainter.paint(
      canvas,
      Offset(
        point.dx + 4.0 / scale,
        point.dy - (textPainter.height + 2.0 / scale),
      ),
    );
  }

  @override
  bool shouldRepaint(covariant SegmentationDisplayPainter oldDelegate) {
    return oldDelegate.masks != masks ||
        oldDelegate.selectedMaskId != selectedMaskId ||
        oldDelegate.scale != scale ||
        oldDelegate.currentProcessingMask != currentProcessingMask;
  }
}

// Main segmentation component with image display
class SegmentationImageComponent extends StatefulWidget {
  final Size imageSize;
  final TransformationController transformationController;
  final Function(List<SegmentationMask>) onSegmentationsChanged;
  final String apiBaseUrl;
  final dynamic originalImageFile; // Either File or Uint8List

  const SegmentationImageComponent({
    Key? key,
    required this.imageSize,
    required this.transformationController,
    required this.onSegmentationsChanged,
    required this.apiBaseUrl,
    required this.originalImageFile,
  }) : super(key: key);

  @override
  State<SegmentationImageComponent> createState() =>
      _SegmentationImageComponentState();
}

class _SegmentationImageComponentState
    extends State<SegmentationImageComponent> {
  // List of segmentation masks
  final List<SegmentationMask> _masks = [];

  // Currently selected mask ID
  String? _selectedMaskId;

  // Current scale from transformation
  double _currentScale = 1.0;

  // Processing state
  bool _isProcessing = false;
  Uint8List? _currentMaskImage; // Result from FastSAM while processing

  // Type of point (foreground or background)
  bool _isForegroundPoint = true;

  // Confidence and IoU thresholds
  double _confidenceThreshold = 0.4;
  double _iouThreshold = 0.9;

  // For displaying image masks
  Map<String, ui.Image?> _maskImagesCache = {};

  @override
  void initState() {
    super.initState();
    // Listen for transformation changes
    widget.transformationController.addListener(_updateScale);
    _updateScale();
  }

  @override
  void dispose() {
    widget.transformationController.removeListener(_updateScale);
    super.dispose();
  }

  // Update current scale from transformation controller
  void _updateScale() {
    final scale = widget.transformationController.value.getMaxScaleOnAxis();
    setState(() {
      _currentScale = scale;
    });
  }

  // Convert screen coordinates to image coordinates
  Offset _screenToImageCoordinates(Offset screenPoint) {
    // Get the current transformation
    final Matrix4 transform = widget.transformationController.value;

    // Calculate the translation part of the transform
    final translation = Offset(
      transform.getTranslation().x,
      transform.getTranslation().y,
    );

    // Adjust point by translation and scale
    final adjustedPoint = (screenPoint - translation) / _currentScale;

    return adjustedPoint;
  }

  // Process the click to get segmentation from FastSAM API
  Future<void> _processClickSegmentation(
    Offset clickPoint,
    bool isForeground,
  ) async {
    if (_isProcessing) return;

    setState(() {
      _isProcessing = true;
      _currentMaskImage = null;
    });

    try {
      // Get the API URL based on platform
      String apiUrl = '${widget.apiBaseUrl}/segment_click/';

      // Create multipart request for the API endpoint
      final request = http.MultipartRequest('POST', Uri.parse(apiUrl));

      // Add image data
      if (widget.originalImageFile != null) {
        if (kIsWeb) {
          if (widget.originalImageFile is Uint8List) {
            request.files.add(
              http.MultipartFile.fromBytes(
                'image',
                widget.originalImageFile,
                filename: 'image.jpg',
              ),
            );
          }
        } else {
          if (widget.originalImageFile is File) {
            request.files.add(
              await http.MultipartFile.fromPath(
                'image',
                widget.originalImageFile.path,
              ),
            );
          }
        }
      }

      // Add click coordinates and parameters
      request.fields['point_x'] = clickPoint.dx.toString();
      request.fields['point_y'] = clickPoint.dy.toString();
      request.fields['point_type'] = isForeground ? '1' : '0';
      request.fields['conf'] = _confidenceThreshold.toString();
      request.fields['iou'] = _iouThreshold.toString();

      // Send the request
      final streamedResponse = await request.send();

      if (streamedResponse.statusCode != 200) {
        final responseString = await streamedResponse.stream.bytesToString();
        throw Exception(
          'Failed to segment image: ${streamedResponse.statusCode} - $responseString',
        );
      }

      // Get response bytes for the mask image
      final responseBytes = await streamedResponse.stream.toBytes();

      // Create a new SegmentationMask with the mask image
      final maskId = 'mask_${DateTime.now().millisecondsSinceEpoch}';

      final newMask = SegmentationMask(
        id: maskId,
        clickPoint: clickPoint,
        maskImage: responseBytes,
        isForeground: isForeground,
      );

      // Add mask to the list
      setState(() {
        _masks.add(newMask);
        _selectedMaskId = maskId;
        _currentMaskImage = null; // Clear temporary mask image
        _isProcessing = false;
      });

      // Notify parent about changes
      widget.onSegmentationsChanged(_masks);

      // Show annotation dialog
      _showAnnotationDialog(newMask);
    } catch (e) {
      print('Error in segmentation: $e');
      setState(() {
        _isProcessing = false;
        _currentMaskImage = null;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to process segmentation: ${e.toString()}'),
        ),
      );
    }
  }

  // Find the mask at the given position
  String? _getMaskIdAtPosition(Offset position) {
    // Simple proximity check to click point - in a real implementation
    // you would check if the point is inside the mask using pixel data
    const hitDistance = 20.0;

    // Check from last to first (top-most first)
    for (int i = _masks.length - 1; i >= 0; i--) {
      if ((position - _masks[i].clickPoint).distance < hitDistance) {
        return _masks[i].id;
      }
    }
    return null;
  }

  // Delete the selected mask
  void _deleteSelectedMask() {
    if (_selectedMaskId == null) return;

    setState(() {
      _masks.removeWhere((mask) => mask.id == _selectedMaskId);
      _selectedMaskId = null;
    });

    // Notify parent about changes
    widget.onSegmentationsChanged(_masks);
  }

  // Show dialog to annotate the selected mask
  void _showAnnotationDialog(SegmentationMask mask) {
    // Create the controller locally
    final TextEditingController noteController = TextEditingController(
      text: mask.note,
    );

    // Track if dialog is active to prevent multiple actions
    bool dialogActive = true;

    showDialog(
      context: context,
      barrierDismissible: true,
      builder: (BuildContext dialogContext) {
        return AlertDialog(
          title: const Text('Add Annotation'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                controller: noteController,
                decoration: const InputDecoration(
                  labelText: 'Note',
                  hintText: 'Enter a description for this segment',
                  border: OutlineInputBorder(),
                ),
                maxLines: 3,
                autofocus: true,
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () {
                if (!dialogActive) return;
                // Mark dialog as inactive before any actions
                dialogActive = false;
                // Close dialog first
                Navigator.of(dialogContext).pop();
                // Safely dispose controller
                noteController.dispose();
              },
              child: const Text('Cancel'),
            ),
            TextButton(
              onPressed: () {
                if (!dialogActive) return;
                // Mark dialog as inactive before any actions
                dialogActive = false;
                // Get the note before disposing controller
                final note = noteController.text.trim();

                // Update the note for the selected mask
                final maskIndex = _masks.indexWhere((m) => m.id == mask.id);
                if (maskIndex != -1) {
                  setState(() {
                    _masks[maskIndex].note = note;
                  });

                  widget.onSegmentationsChanged(_masks);
                }

                // Close dialog first
                Navigator.of(dialogContext).pop();
                // Safely dispose controller
                noteController.dispose();
              },
              child: const Text('Save'),
            ),
          ],
        );
      },
    );
  }

  // Show info about all segmentation masks
  void _showAllMasksInfo() {
    showDialog(
      context: context,
      builder:
          (context) => AlertDialog(
            title: const Text('Segmentation Masks'),
            content: SizedBox(
              width: double.maxFinite,
              child: ListView.builder(
                shrinkWrap: true,
                itemCount: _masks.length,
                itemBuilder: (context, index) {
                  final mask = _masks[index];
                  return ListTile(
                    title: Text(mask.note.isNotEmpty ? mask.note : 'No note'),
                    subtitle: Text(
                      'Click X: ${mask.clickPoint.dx.toStringAsFixed(1)}, Y: ${mask.clickPoint.dy.toStringAsFixed(1)}, '
                      'Type: ${mask.isForeground ? 'Foreground' : 'Background'}',
                    ),
                    leading: Icon(
                      mask.isForeground
                          ? Icons.add_circle
                          : Icons.remove_circle,
                      color:
                          mask.id == _selectedMaskId
                              ? Colors.green
                              : mask.color,
                    ),
                    onTap: () {
                      // Select this mask and close the dialog
                      setState(() {
                        _selectedMaskId = mask.id;
                      });
                      Navigator.of(context).pop();
                    },
                  );
                },
              ),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text('Close'),
              ),
            ],
          ),
    );
  }

  // Widget to render a segmentation mask
  Widget _buildMaskDisplayLayer() {
    return CustomPaint(
      painter: SegmentationDisplayPainter(
        masks: _masks,
        selectedMaskId: _selectedMaskId,
        scale: _currentScale,
        currentProcessingMask: _currentMaskImage,
      ),
      size: widget.imageSize,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // ToggleButtons for point type selection
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 8.0),
          child: ToggleButtons(
            isSelected: [_isForegroundPoint, !_isForegroundPoint],
            onPressed: (index) {
              setState(() {
                _isForegroundPoint = index == 0;
              });
            },
            children: const [
              Padding(
                padding: EdgeInsets.symmetric(horizontal: 16.0),
                child: Text('Foreground'),
              ),
              Padding(
                padding: EdgeInsets.symmetric(horizontal: 16.0),
                child: Text('Background'),
              ),
            ],
          ),
        ),

        Expanded(
          child: MouseRegion(
            cursor:
                _isProcessing
                    ? SystemMouseCursors.wait
                    : SystemMouseCursors.click,
            child: GestureDetector(
              onTapDown: (details) {
                if (_isProcessing) return;

                final imagePoint = _screenToImageCoordinates(
                  details.localPosition,
                );

                // Check if clicking inside an existing mask
                final maskId = _getMaskIdAtPosition(imagePoint);
                if (maskId != null) {
                  setState(() {
                    _selectedMaskId = maskId;
                  });
                  return;
                }

                // Process a new segmentation
                _processClickSegmentation(imagePoint, _isForegroundPoint);
              },
              onDoubleTap: () {
                if (_selectedMaskId != null) {
                  // Find the mask and show the annotation dialog
                  final mask = _masks.firstWhere(
                    (mask) => mask.id == _selectedMaskId,
                    orElse: () => _masks.first,
                  );
                  _showAnnotationDialog(mask);
                }
              },
              child: Stack(
                children: [
                  // Original image layer is drawn by parent widget

                  // Segmentation masks layer
                  _buildMaskDisplayLayer(),

                  // Processing indicator
                  if (_isProcessing)
                    const Center(child: CircularProgressIndicator()),

                  // Floating action buttons
                  Positioned(
                    right: 16,
                    bottom: 16,
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        if (_masks.isNotEmpty)
                          FloatingActionButton.small(
                            heroTag: 'info',
                            child: const Icon(Icons.info_outline),
                            onPressed: _showAllMasksInfo,
                            tooltip: 'Show all segments',
                          ),
                        const SizedBox(height: 8),
                        if (_selectedMaskId != null)
                          FloatingActionButton.small(
                            heroTag: 'edit',
                            child: const Icon(Icons.edit),
                            onPressed: () {
                              final mask = _masks.firstWhere(
                                (mask) => mask.id == _selectedMaskId,
                              );
                              _showAnnotationDialog(mask);
                            },
                            tooltip: 'Edit segment',
                          ),
                        const SizedBox(height: 8),
                        if (_selectedMaskId != null)
                          FloatingActionButton.small(
                            heroTag: 'delete',
                            backgroundColor: Colors.red,
                            child: const Icon(Icons.delete),
                            onPressed: _deleteSelectedMask,
                            tooltip: 'Delete segment',
                          ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),

        // Threshold sliders
        Padding(
          padding: const EdgeInsets.all(8.0),
          child: Column(
            children: [
              Row(
                children: [
                  const Text('Confidence:'),
                  Expanded(
                    child: Slider(
                      value: _confidenceThreshold,
                      min: 0.1,
                      max: 0.9,
                      divisions: 8,
                      label: _confidenceThreshold.toStringAsFixed(1),
                      onChanged: (value) {
                        setState(() {
                          _confidenceThreshold = value;
                        });
                      },
                    ),
                  ),
                  Text(_confidenceThreshold.toStringAsFixed(1)),
                ],
              ),
              Row(
                children: [
                  const Text('IoU:'),
                  Expanded(
                    child: Slider(
                      value: _iouThreshold,
                      min: 0.5,
                      max: 0.95,
                      divisions: 9,
                      label: _iouThreshold.toStringAsFixed(2),
                      onChanged: (value) {
                        setState(() {
                          _iouThreshold = value;
                        });
                      },
                    ),
                  ),
                  Text(_iouThreshold.toStringAsFixed(2)),
                ],
              ),
            ],
          ),
        ),
      ],
    );
  }
}
