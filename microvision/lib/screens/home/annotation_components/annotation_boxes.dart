import 'dart:math' as math;

import 'package:flutter/material.dart';

// Model class for annotation data
class AnnotationBox {
  final String id;
  Rect bounds;
  String note;

  AnnotationBox({required this.id, required this.bounds, this.note = ''});

  // Convert to map for serialization
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'x': bounds.left,
      'y': bounds.top,
      'width': bounds.width,
      'height': bounds.height,
      'note': note,
    };
  }
}

// Custom painter for drawing annotation boxes
class AnnotationBoxPainter extends CustomPainter {
  final List<AnnotationBox> boxes;
  final String? selectedBoxId;
  final double scale;

  AnnotationBoxPainter({
    required this.boxes,
    this.selectedBoxId,
    this.scale = 1.0,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Define styles for different box states
    final Paint normalBoxPaint =
        Paint()
          ..color = Colors.blue.withOpacity(0.3)
          ..strokeWidth = 2.0 / scale
          ..style = PaintingStyle.fill;

    final Paint selectedBoxPaint =
        Paint()
          ..color = Colors.green.withOpacity(0.3)
          ..strokeWidth = 2.0 / scale
          ..style = PaintingStyle.fill;

    final Paint normalBorderPaint =
        Paint()
          ..color = Colors.blue
          ..strokeWidth = 2.0 / scale
          ..style = PaintingStyle.stroke;

    final Paint selectedBorderPaint =
        Paint()
          ..color = Colors.green
          ..strokeWidth = 2.0 / scale
          ..style = PaintingStyle.stroke;

    // Draw all boxes
    for (final box in boxes) {
      final bool isSelected = box.id == selectedBoxId;

      // Fill
      canvas.drawRect(
        box.bounds,
        isSelected ? selectedBoxPaint : normalBoxPaint,
      );

      // Border
      canvas.drawRect(
        box.bounds,
        isSelected ? selectedBorderPaint : normalBorderPaint,
      );

      // Draw resize handles if selected
      if (isSelected) {
        _drawResizeHandles(canvas, box.bounds, scale);
      }

      // Draw note label if not empty
      if (box.note.isNotEmpty) {
        _drawNoteLabel(canvas, box.bounds, box.note, scale);
      }
    }
  }

  void _drawResizeHandles(Canvas canvas, Rect bounds, double scale) {
    final handlePaint =
        Paint()
          ..color = Colors.white
          ..strokeWidth = 1.0 / scale
          ..style = PaintingStyle.fill;

    final handleBorderPaint =
        Paint()
          ..color = Colors.green
          ..strokeWidth = 1.0 / scale
          ..style = PaintingStyle.stroke;

    // Handle size adjusted for scale
    final handleSize = 8.0 / scale;

    // Positions for the 8 handles
    final positions = [
      Offset(bounds.left, bounds.top), // Top-left
      Offset(bounds.left + bounds.width / 2, bounds.top), // Top-center
      Offset(bounds.right, bounds.top), // Top-right
      Offset(bounds.right, bounds.top + bounds.height / 2), // Middle-right
      Offset(bounds.right, bounds.bottom), // Bottom-right
      Offset(bounds.left + bounds.width / 2, bounds.bottom), // Bottom-center
      Offset(bounds.left, bounds.bottom), // Bottom-left
      Offset(bounds.left, bounds.top + bounds.height / 2), // Middle-left
    ];

    // Draw handles
    for (final position in positions) {
      final handleRect = Rect.fromCenter(
        center: position,
        width: handleSize,
        height: handleSize,
      );

      canvas.drawRect(handleRect, handlePaint);
      canvas.drawRect(handleRect, handleBorderPaint);
    }
  }

  void _drawNoteLabel(Canvas canvas, Rect bounds, String note, double scale) {
    final textSpan = TextSpan(
      text: note,
      style: TextStyle(
        color: Colors.white,
        fontSize: 14.0 / scale,
        fontWeight: FontWeight.bold,
      ),
    );

    final textPainter = TextPainter(
      text: textSpan,
      textDirection: TextDirection.ltr,
    );

    textPainter.layout();

    // Background for text
    final textBackgroundRect = Rect.fromLTWH(
      bounds.left,
      bounds.top - (textPainter.height + 4.0 / scale),
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
        bounds.left + 4.0 / scale,
        bounds.top - (textPainter.height + 2.0 / scale),
      ),
    );
  }

  @override
  bool shouldRepaint(covariant AnnotationBoxPainter oldDelegate) {
    return oldDelegate.boxes != boxes ||
        oldDelegate.selectedBoxId != selectedBoxId ||
        oldDelegate.scale != scale;
  }
}

// Main annotation component
class AnnotationBoxComponent extends StatefulWidget {
  final Size imageSize;
  final TransformationController transformationController;
  final Function(List<AnnotationBox>) onAnnotationsChanged;

  const AnnotationBoxComponent({
    Key? key,
    required this.imageSize,
    required this.transformationController,
    required this.onAnnotationsChanged,
  }) : super(key: key);

  @override
  State<AnnotationBoxComponent> createState() => _AnnotationBoxComponentState();
}

class _AnnotationBoxComponentState extends State<AnnotationBoxComponent> {
  // List of annotation boxes
  final List<AnnotationBox> _boxes = [];

  // Currently selected box ID
  String? _selectedBoxId;

  // Drawing state variables
  bool _isDrawing = false;
  Offset? _startPoint;
  Offset? _currentPoint;

  // Moving and resizing state
  bool _isMoving = false;
  bool _isResizing = false;
  int _resizeHandleIndex = -1;
  Offset? _lastPointerPosition;

  // Current scale from transformation
  double _currentScale = 1.0;

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

  // Check if a point is near a resize handle of the selected box
  int _getResizeHandleIndex(Offset point) {
    if (_selectedBoxId == null) return -1;

    final selectedBox = _boxes.firstWhere((box) => box.id == _selectedBoxId);
    final bounds = selectedBox.bounds;

    // Handle positions (same order as in the painter)
    final positions = [
      Offset(bounds.left, bounds.top), // Top-left
      Offset(bounds.left + bounds.width / 2, bounds.top), // Top-center
      Offset(bounds.right, bounds.top), // Top-right
      Offset(bounds.right, bounds.top + bounds.height / 2), // Middle-right
      Offset(bounds.right, bounds.bottom), // Bottom-right
      Offset(bounds.left + bounds.width / 2, bounds.bottom), // Bottom-center
      Offset(bounds.left, bounds.bottom), // Bottom-left
      Offset(bounds.left, bounds.top + bounds.height / 2), // Middle-left
    ];

    // Check distance to each handle
    final hitDistance = 12.0 / _currentScale; // Adjusted for scale

    for (int i = 0; i < positions.length; i++) {
      if ((positions[i] - point).distance < hitDistance) {
        return i;
      }
    }

    return -1;
  }

  // Find the box at the given position
  String? _getBoxIdAtPosition(Offset position) {
    // Check from last to first (top-most first)
    for (int i = _boxes.length - 1; i >= 0; i--) {
      if (_boxes[i].bounds.contains(position)) {
        return _boxes[i].id;
      }
    }
    return null;
  }

  // Add a new box
  void _addBox(Rect bounds) {
    final String newId = 'box_${DateTime.now().millisecondsSinceEpoch}';

    final newBox = AnnotationBox(id: newId, bounds: bounds);

    setState(() {
      _boxes.add(newBox);
      _selectedBoxId = newId;
      _isDrawing = false;
      _startPoint = null;
      _currentPoint = null;
    });

    // Notify parent about changes
    widget.onAnnotationsChanged(_boxes);

    // Show annotation dialog
    _showAnnotationDialog(newBox);
  }

  // Resize the selected box
  void _resizeSelectedBox(Offset delta, int handleIndex) {
    if (_selectedBoxId == null) return;

    final boxIndex = _boxes.indexWhere((box) => box.id == _selectedBoxId);
    if (boxIndex == -1) return;

    final box = _boxes[boxIndex];
    Rect newBounds = box.bounds;

    // Apply appropriate resize based on which handle was dragged
    switch (handleIndex) {
      case 0: // Top-left
        newBounds = Rect.fromLTRB(
          box.bounds.left + delta.dx,
          box.bounds.top + delta.dy,
          box.bounds.right,
          box.bounds.bottom,
        );
        break;
      case 1: // Top-center
        newBounds = Rect.fromLTRB(
          box.bounds.left,
          box.bounds.top + delta.dy,
          box.bounds.right,
          box.bounds.bottom,
        );
        break;
      case 2: // Top-right
        newBounds = Rect.fromLTRB(
          box.bounds.left,
          box.bounds.top + delta.dy,
          box.bounds.right + delta.dx,
          box.bounds.bottom,
        );
        break;
      case 3: // Middle-right
        newBounds = Rect.fromLTRB(
          box.bounds.left,
          box.bounds.top,
          box.bounds.right + delta.dx,
          box.bounds.bottom,
        );
        break;
      case 4: // Bottom-right
        newBounds = Rect.fromLTRB(
          box.bounds.left,
          box.bounds.top,
          box.bounds.right + delta.dx,
          box.bounds.bottom + delta.dy,
        );
        break;
      case 5: // Bottom-center
        newBounds = Rect.fromLTRB(
          box.bounds.left,
          box.bounds.top,
          box.bounds.right,
          box.bounds.bottom + delta.dy,
        );
        break;
      case 6: // Bottom-left
        newBounds = Rect.fromLTRB(
          box.bounds.left + delta.dx,
          box.bounds.top,
          box.bounds.right,
          box.bounds.bottom + delta.dy,
        );
        break;
      case 7: // Middle-left
        newBounds = Rect.fromLTRB(
          box.bounds.left + delta.dx,
          box.bounds.top,
          box.bounds.right,
          box.bounds.bottom,
        );
        break;
    }

    // Ensure the box isn't flipped (width and height are positive)
    if (newBounds.width <= 0 || newBounds.height <= 0) return;

    // Update the box
    setState(() {
      _boxes[boxIndex] = AnnotationBox(
        id: box.id,
        bounds: newBounds,
        note: box.note,
      );
    });

    // Notify parent about changes
    widget.onAnnotationsChanged(_boxes);
  }

  // Move the selected box
  void _moveSelectedBox(Offset delta) {
    if (_selectedBoxId == null) return;

    final boxIndex = _boxes.indexWhere((box) => box.id == _selectedBoxId);
    if (boxIndex == -1) return;

    final box = _boxes[boxIndex];

    // Create a new bounds with the delta applied
    final newBounds = box.bounds.translate(delta.dx, delta.dy);

    // Update the box
    setState(() {
      _boxes[boxIndex] = AnnotationBox(
        id: box.id,
        bounds: newBounds,
        note: box.note,
      );
    });

    // Notify parent about changes
    widget.onAnnotationsChanged(_boxes);
  }

  // Delete the selected box
  void _deleteSelectedBox() {
    if (_selectedBoxId == null) return;

    setState(() {
      _boxes.removeWhere((box) => box.id == _selectedBoxId);
      _selectedBoxId = null;
    });

    // Notify parent about changes
    widget.onAnnotationsChanged(_boxes);
  }

  // Show dialog to annotate the selected box
  void _showAnnotationDialog(AnnotationBox box) {
    // Create the controller locally
    final TextEditingController noteController = TextEditingController(
      text: box.note,
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
                  hintText: 'Enter a description for this region',
                  border: OutlineInputBorder(),
                ),
                maxLines: 3,
                autofocus: true,
              ),
              // Other content...
            ],
          ),
          actions: [
            TextButton(
              onPressed: () {
                if (!dialogActive) return;
                // Mark dialog as inactive before any actions
                dialogActive = false;
                // Get the note before disposing controller
                final note = noteController.text.trim();
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

                // Update the note for the selected box
                final boxIndex = _boxes.indexWhere((b) => b.id == box.id);
                if (boxIndex != -1) {
                  setState(() {
                    _boxes[boxIndex] = AnnotationBox(
                      id: box.id,
                      bounds: box.bounds,
                      note: note,
                    );
                  });

                  widget.onAnnotationsChanged(_boxes);
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

  // Show info about multiple annotation boxes
  void _showMultiBoxInfo() {
    showDialog(
      context: context,
      builder:
          (context) => AlertDialog(
            title: const Text('Annotation Boxes'),
            content: SizedBox(
              width: double.maxFinite,
              child: ListView.builder(
                shrinkWrap: true,
                itemCount: _boxes.length,
                itemBuilder: (context, index) {
                  final box = _boxes[index];
                  return ListTile(
                    title: Text(box.note.isNotEmpty ? box.note : 'No note'),
                    subtitle: Text(
                      'X: ${box.bounds.left.toStringAsFixed(1)}, Y: ${box.bounds.top.toStringAsFixed(1)}, '
                      'W: ${box.bounds.width.toStringAsFixed(1)}, H: ${box.bounds.height.toStringAsFixed(1)}',
                    ),
                    leading: Icon(
                      Icons.crop_square,
                      color:
                          box.id == _selectedBoxId ? Colors.green : Colors.blue,
                    ),
                    onTap: () {
                      // Select this box and close the dialog
                      setState(() {
                        _selectedBoxId = box.id;
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

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      cursor:
          _isDrawing || _isMoving || _isResizing
              ? SystemMouseCursors.grabbing
              : SystemMouseCursors.precise,
      child: GestureDetector(
        onPanStart: (details) {
          final imagePoint = _screenToImageCoordinates(details.localPosition);

          // Check if clicking on a resize handle
          if (_selectedBoxId != null) {
            final handleIndex = _getResizeHandleIndex(imagePoint);
            if (handleIndex != -1) {
              setState(() {
                _isResizing = true;
                _resizeHandleIndex = handleIndex;
                _lastPointerPosition = imagePoint;
              });
              return;
            }
          }

          // Check if clicking inside an existing box
          final boxId = _getBoxIdAtPosition(imagePoint);
          if (boxId != null) {
            setState(() {
              _selectedBoxId = boxId;
              _isMoving = true;
              _lastPointerPosition = imagePoint;
            });
            return;
          }

          // Start drawing a new box
          setState(() {
            _isDrawing = true;
            _startPoint = imagePoint;
            _currentPoint = imagePoint;
            _selectedBoxId = null;
          });
        },
        onPanUpdate: (details) {
          final imagePoint = _screenToImageCoordinates(details.localPosition);

          if (_isResizing &&
              _selectedBoxId != null &&
              _lastPointerPosition != null) {
            // Handle resize
            final delta = imagePoint - _lastPointerPosition!;
            _resizeSelectedBox(delta, _resizeHandleIndex);
            setState(() {
              _lastPointerPosition = imagePoint;
            });
          } else if (_isMoving &&
              _selectedBoxId != null &&
              _lastPointerPosition != null) {
            // Handle move
            final delta = imagePoint - _lastPointerPosition!;
            _moveSelectedBox(delta);
            setState(() {
              _lastPointerPosition = imagePoint;
            });
          } else if (_isDrawing && _startPoint != null) {
            // Handle drawing
            setState(() {
              _currentPoint = imagePoint;
            });
          }
        },
        onPanEnd: (details) {
          if (_isDrawing && _startPoint != null && _currentPoint != null) {
            // Finalize drawing
            final left = math.min(_startPoint!.dx, _currentPoint!.dx);
            final top = math.min(_startPoint!.dy, _currentPoint!.dy);
            final width = (_startPoint!.dx - _currentPoint!.dx).abs();
            final height = (_startPoint!.dy - _currentPoint!.dy).abs();

            // Only create a box if it has a reasonable size
            if (width > 10 / _currentScale && height > 10 / _currentScale) {
              _addBox(Rect.fromLTWH(left, top, width, height));
            }
          }

          setState(() {
            _isDrawing = false;
            _isMoving = false;
            _isResizing = false;
            _lastPointerPosition = null;
          });
        },
        onTap: () {
          final imagePoint = _screenToImageCoordinates(
            Offset(
              (context.findRenderObject() as RenderBox)
                  .globalToLocal(Offset.zero)
                  .dx,
              (context.findRenderObject() as RenderBox)
                  .globalToLocal(Offset.zero)
                  .dy,
            ),
          );

          final boxId = _getBoxIdAtPosition(imagePoint);

          setState(() {
            _selectedBoxId = boxId;
          });
        },
        onDoubleTap: () {
          if (_selectedBoxId != null) {
            // Find the box and show the annotation dialog
            final box = _boxes.firstWhere(
              (box) => box.id == _selectedBoxId,
              orElse: () => _boxes.first,
            );
            _showAnnotationDialog(box);
          }
        },
        child: CustomPaint(
          painter: AnnotationBoxPainter(
            boxes: _boxes,
            selectedBoxId: _selectedBoxId,
            scale: _currentScale,
          ),
          child: Stack(
            children: [
              // Drawing preview
              if (_isDrawing && _startPoint != null && _currentPoint != null)
                CustomPaint(
                  painter: AnnotationBoxPainter(
                    boxes: [
                      AnnotationBox(
                        id: 'preview',
                        bounds: Rect.fromPoints(_startPoint!, _currentPoint!),
                      ),
                    ],
                    scale: _currentScale,
                  ),
                ),

              // Floating action buttons
              Positioned(
                right: 16,
                bottom: 16,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (_boxes.isNotEmpty)
                      FloatingActionButton.small(
                        heroTag: 'info',
                        child: const Icon(Icons.info_outline),
                        onPressed: _showMultiBoxInfo,
                        tooltip: 'Show all annotations',
                      ),
                    const SizedBox(height: 8),
                    if (_selectedBoxId != null)
                      FloatingActionButton.small(
                        heroTag: 'edit',
                        child: const Icon(Icons.edit),
                        onPressed: () {
                          final box = _boxes.firstWhere(
                            (box) => box.id == _selectedBoxId,
                          );
                          _showAnnotationDialog(box);
                        },
                        tooltip: 'Edit annotation',
                      ),
                    const SizedBox(height: 8),
                    if (_selectedBoxId != null)
                      FloatingActionButton.small(
                        heroTag: 'delete',
                        backgroundColor: Colors.red,
                        child: const Icon(Icons.delete),
                        onPressed: _deleteSelectedBox,
                        tooltip: 'Delete annotation',
                      ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
