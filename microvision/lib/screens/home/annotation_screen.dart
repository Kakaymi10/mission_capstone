import 'package:flutter/material.dart';
import 'dart:io';
import 'dart:typed_data';
import 'dart:convert';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:microvision/screens/home/annotation_components/annotation_boxes.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class AnnotationScreen extends StatefulWidget {
  final dynamic imageFile; // Can be File on mobile or Uint8List on web
  final Function(dynamic) onComplete; // Returns annotated image

  const AnnotationScreen({
    Key? key,
    required this.imageFile,
    required this.onComplete,
  }) : super(key: key);

  @override
  State<AnnotationScreen> createState() => _AnnotationScreenState();
}

class _AnnotationScreenState extends State<AnnotationScreen> {
  // Controller for the transform
  final TransformationController _transformationController =
      TransformationController();

  // Track the current scale
  double _currentScale = 1.0;

  // Min and max scale constraints
  final double _minScale = 0.5;
  final double _maxScale = 3.0;

  // Store image dimensions
  Size _imageSize = Size.zero;

  // Store annotations
  List<AnnotationBox> _annotations = [];

  // Mode selection (pan/zoom vs annotate)
  bool _isAnnotationMode = true;

  // Supabase client instance

  @override
  void initState() {
    super.initState();
    // Initialize with identity matrix (no transformation)
    _transformationController.value = Matrix4.identity();

    // Get image dimensions as soon as possible
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _getImageDimensions();
    });
  }

  // Get image dimensions as soon as possible
  Future<void> _getImageDimensions() async {
    try {
      if (kIsWeb) {
        if (widget.imageFile is Uint8List) {
          final image = await decodeImageFromList(
            widget.imageFile as Uint8List,
          );
          setState(() {
            _imageSize = Size(image.width.toDouble(), image.height.toDouble());
          });
        }
      } else {
        if (widget.imageFile is File) {
          final bytes = await (widget.imageFile as File).readAsBytes();
          final image = await decodeImageFromList(bytes);
          setState(() {
            _imageSize = Size(image.width.toDouble(), image.height.toDouble());
          });
        }
      }
    } catch (e) {
      print('Error getting image dimensions: $e');
    }
  }

  @override
  void dispose() {
    _transformationController.dispose();
    super.dispose();
  }

  // Reset zoom and position
  void _resetTransformation() {
    setState(() {
      _transformationController.value = Matrix4.identity();
      _currentScale = 1.0;
    });
  }

  // Build the image widget based on platform
  Widget _buildImageWidget() {
    if (kIsWeb) {
      if (widget.imageFile is Uint8List) {
        return Image.memory(widget.imageFile as Uint8List, fit: BoxFit.contain);
      }
    } else {
      if (widget.imageFile is File) {
        return Image.file(widget.imageFile as File, fit: BoxFit.contain);
      }
    }

    // Fallback
    return const Center(child: Text('Invalid image format'));
  }

  // Update the annotations list
  void _updateAnnotations(List<AnnotationBox> annotations) {
    setState(() {
      _annotations = annotations;
    });
  }

  // Convert annotations to JSON
  List<Map<String, dynamic>> _getAnnotationsAsJson() {
    return _annotations.map((box) => box.toJson()).toList();
  }

  // Get normalized annotations (coordinates relative to image size)
  List<Map<String, dynamic>> _getNormalizedAnnotations() {
    return _annotations.map((box) {
      final normalizedBox = Map<String, dynamic>.from(box.toJson());
      normalizedBox['x'] = normalizedBox['x'] / _imageSize.width;
      normalizedBox['y'] = normalizedBox['y'] / _imageSize.height;
      normalizedBox['width'] = normalizedBox['width'] / _imageSize.width;
      normalizedBox['height'] = normalizedBox['height'] / _imageSize.height;
      return normalizedBox;
    }).toList();
  }

  // Generate a safe image ID
  String _generateImageId() {
    if (widget.imageFile is File) {
      // For File objects, create a hash of the path
      final path = (widget.imageFile as File).path;
      final pathHash = path.hashCode.toString();
      return 'file_$pathHash';
    } else if (kIsWeb && widget.imageFile is Uint8List) {
      // For web, create a timestamp-based ID
      return 'web_${DateTime.now().millisecondsSinceEpoch}';
    }
    // Fallback
    return 'image_${DateTime.now().millisecondsSinceEpoch}';
  }

  // Show a loading dialog

  void _completeAnnotation() async {
    // First, check if we have annotations
    if (_annotations.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('No annotations created. Draw at least one box.'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    try {
      // Generate an image ID
      final String imageId = _generateImageId();

      // Prepare the annotation data
      final Map<String, dynamic> data = {
        'success': true,
        'id': imageId,
        'annotations': jsonEncode(_getNormalizedAnnotations()),
        'raw_annotations': jsonEncode(_getAnnotationsAsJson()),
        'image_width': _imageSize.width,
        'image_height': _imageSize.height,
      };

      // Pass data via callback, but don't immediately pop
      // Schedule the navigation to happen after the current frame is complete
      WidgetsBinding.instance.addPostFrameCallback((_) {
        // First call the callback
        widget.onComplete(data);

        // Then after a slight delay, pop the screen
        Future.delayed(Duration(milliseconds: 100), () {
          if (mounted) {
            Navigator.of(context).pop();
          }
        });
      });
    } catch (e) {
      // Show error message
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error preparing annotations: ${e.toString()}'),
          backgroundColor: Colors.red,
        ),
      );

      // Call the callback with failure after current frame
      WidgetsBinding.instance.addPostFrameCallback((_) {
        widget.onComplete({'success': false});

        // Delay popping
        Future.delayed(Duration(milliseconds: 100), () {
          if (mounted) {
            Navigator.of(context).pop();
          }
        });
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Annotate Image'),
        actions: [
          // Toggle between annotation and pan mode
          IconButton(
            icon: Icon(_isAnnotationMode ? Icons.pan_tool : Icons.edit),
            tooltip:
                _isAnnotationMode
                    ? 'Switch to Pan Mode'
                    : 'Switch to Annotation Mode',
            onPressed: () {
              setState(() {
                _isAnnotationMode = !_isAnnotationMode;
              });
            },
          ),
          // Reset button
          IconButton(
            icon: const Icon(Icons.refresh),
            tooltip: 'Reset View',
            onPressed: _resetTransformation,
          ),
          // Info button to show annotation data
          IconButton(
            icon: const Icon(Icons.info_outline),
            tooltip: 'Show Annotation Data',
            onPressed:
                _annotations.isEmpty
                    ? null
                    : () {
                      // Show a dialog with annotation data
                      showDialog(
                        context: context,
                        builder:
                            (context) => AlertDialog(
                              title: const Text('Annotation Data'),
                              content: SingleChildScrollView(
                                child: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      'Number of annotations: ${_annotations.length}',
                                    ),
                                    const SizedBox(height: 10),
                                    const Text(
                                      'Coordinates (normalized to image size):',
                                    ),
                                    const SizedBox(height: 5),
                                    for (final box in _annotations)
                                      Padding(
                                        padding: const EdgeInsets.only(
                                          bottom: 8.0,
                                        ),
                                        child: Card(
                                          child: Padding(
                                            padding: const EdgeInsets.all(8.0),
                                            child: Column(
                                              crossAxisAlignment:
                                                  CrossAxisAlignment.start,
                                              children: [
                                                Text(
                                                  'Label: ${box.note.isNotEmpty ? box.note : "No label"}',
                                                  style: const TextStyle(
                                                    fontWeight: FontWeight.bold,
                                                  ),
                                                ),
                                                Text(
                                                  'X: ${(box.bounds.left / _imageSize.width).toStringAsFixed(3)}, '
                                                  'Y: ${(box.bounds.top / _imageSize.height).toStringAsFixed(3)}',
                                                ),
                                                Text(
                                                  'Width: ${(box.bounds.width / _imageSize.width).toStringAsFixed(3)}, '
                                                  'Height: ${(box.bounds.height / _imageSize.height).toStringAsFixed(3)}',
                                                ),
                                              ],
                                            ),
                                          ),
                                        ),
                                      ),
                                  ],
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
                    },
          ),
          // Save button
          IconButton(
            icon: const Icon(Icons.check),
            tooltip: 'Complete',
            onPressed: _completeAnnotation,
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: Stack(
              children: [
                InteractiveViewer(
                  transformationController: _transformationController,
                  minScale: _minScale,
                  maxScale: _maxScale,
                  panEnabled: !_isAnnotationMode,
                  scaleEnabled: !_isAnnotationMode,
                  onInteractionEnd: (ScaleEndDetails details) {
                    // Update current scale when interaction ends
                    final scale =
                        _transformationController.value.getMaxScaleOnAxis();
                    setState(() {
                      _currentScale = scale;
                    });
                  },
                  child: Center(
                    child: Stack(
                      children: [
                        // The image
                        _buildImageWidget(),

                        // The annotation layer - only show if image size is known
                        if (_imageSize != Size.zero)
                          SizedBox(
                            width: _imageSize.width,
                            height: _imageSize.height,
                            child:
                                _isAnnotationMode
                                    ? AnnotationBoxComponent(
                                      imageSize: _imageSize,
                                      transformationController:
                                          _transformationController,
                                      onAnnotationsChanged: _updateAnnotations,
                                    )
                                    : const SizedBox.shrink(),
                          ),
                      ],
                    ),
                  ),
                ),

                // Help text overlay
                if (_isAnnotationMode)
                  Positioned(
                    top: 16,
                    left: 0,
                    right: 0,
                    child: Center(
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 8,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.7),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: const Text(
                          'Draw: Drag to create box | Edit: Double-tap box',
                          style: TextStyle(color: Colors.white),
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
          // Bottom controls
          Container(
            padding: const EdgeInsets.all(16),
            color: Colors.grey[200],
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    // Zoom out button
                    FloatingActionButton(
                      heroTag: 'zoomOut',
                      mini: true,
                      child: const Icon(Icons.zoom_out),
                      onPressed: () {
                        final newScale = (_currentScale - 0.25).clamp(
                          _minScale,
                          _maxScale,
                        );
                        final Matrix4 matrix =
                            Matrix4.identity()..scale(newScale, newScale);
                        setState(() {
                          _transformationController.value = matrix;
                          _currentScale = newScale;
                        });
                      },
                    ),
                    // Current scale indicator
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 8),
                      child: Text(
                        '${(_currentScale * 100).round()}%',
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                    ),
                    // Zoom in button
                    FloatingActionButton(
                      heroTag: 'zoomIn',
                      mini: true,
                      child: const Icon(Icons.zoom_in),
                      onPressed: () {
                        final newScale = (_currentScale + 0.25).clamp(
                          _minScale,
                          _maxScale,
                        );
                        final Matrix4 matrix =
                            Matrix4.identity()..scale(newScale, newScale);
                        setState(() {
                          _transformationController.value = matrix;
                          _currentScale = newScale;
                        });
                      },
                    ),
                  ],
                ),

                // Annotation count
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.blue,
                    borderRadius: BorderRadius.circular(15),
                  ),
                  child: Text(
                    '${_annotations.length} annotations',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
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
