// 2. components.dart - Contains all UI components
import 'package:flutter/material.dart';
import 'package:flutter_colorpicker/flutter_colorpicker.dart';
import 'models.dart';
import 'dart:ui' as ui;

// Tool overlay widget
class ToolOverlayWidget extends StatelessWidget {
  final bool isVisible;
  final Color selectedColor;
  final double strokeWidth;
  final List<Color> presetColors;
  final Function(Color) onColorChanged;
  final Function(double) onWidthChanged;

  const ToolOverlayWidget({
    super.key,
    required this.isVisible,
    required this.selectedColor,
    required this.strokeWidth,
    required this.presetColors,
    required this.onColorChanged,
    required this.onWidthChanged,
  });

  @override
  Widget build(BuildContext context) {
    if (!isVisible) return const SizedBox.shrink();

    return Positioned(
      top: 10,
      right: 10,
      child: Card(
        elevation: 8,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        child: Padding(
          padding: const EdgeInsets.all(8.0),
          child: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Color selection
                Wrap(
                  spacing: 5,
                  runSpacing: 5,
                  children:
                      presetColors.map((color) {
                        return GestureDetector(
                          onTap: () => onColorChanged(color),
                          child: Container(
                            width: 30,
                            height: 30,
                            decoration: BoxDecoration(
                              color: color,
                              shape: BoxShape.circle,
                              border: Border.all(
                                color:
                                    selectedColor == color
                                        ? Colors.white
                                        : Colors.black,
                                width: selectedColor == color ? 3 : 1,
                              ),
                              boxShadow:
                                  selectedColor == color
                                      ? [
                                        BoxShadow(
                                          color: Colors.black.withOpacity(0.3),
                                          blurRadius: 4,
                                        ),
                                      ]
                                      : null,
                            ),
                          ),
                        );
                      }).toList(),
                ),
                const SizedBox(height: 10),
                // Custom color picker button
                IconButton(
                  icon: Icon(Icons.color_lens, color: selectedColor),
                  onPressed: () => _showColorPicker(context),
                  tooltip: 'Custom Color',
                ),
                const SizedBox(height: 5),
                // Stroke width slider
                SizedBox(
                  height: 120,
                  width: 40,
                  child: RotatedBox(
                    quarterTurns: 3,
                    child: Slider(
                      value: strokeWidth,
                      min: 1,
                      max: 20,
                      divisions: 19,
                      label: strokeWidth.round().toString(),
                      onChanged: onWidthChanged,
                    ),
                  ),
                ),
                const SizedBox(height: 5),
                // Stroke width preview
                Container(
                  width: strokeWidth,
                  height: strokeWidth,
                  decoration: BoxDecoration(
                    color: selectedColor,
                    shape: BoxShape.circle,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  void _showColorPicker(BuildContext context) {
    showDialog(
      context: context,
      builder:
          (context) => AlertDialog(
            title: const Text('Select Color'),
            content: SingleChildScrollView(
              child: ColorPicker(
                pickerColor: selectedColor,
                onColorChanged: onColorChanged,
                pickerAreaHeightPercent: 0.8,
                enableAlpha: true,
                displayThumbColor: true,
                showLabel: true,
                paletteType: PaletteType.hsv,
              ),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Done'),
              ),
            ],
          ),
    );
  }
}

// Learning panel widget (now horizontal at bottom)
class LearningPanelWidget extends StatelessWidget {
  final Map<String, dynamic> specimen;
  final Animation<double> animation;
  final VoidCallback onClose;

  const LearningPanelWidget({
    super.key,
    required this.specimen,
    required this.animation,
    required this.onClose,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: animation,
      builder: (context, child) {
        return Positioned(
          left: 0,
          right: 0,
          bottom: 0,
          height:
              MediaQuery.of(context).size.height *
              0.8 *
              (1 -
                  animation
                      .value), // Increased from 0.6 to 0.8 (80% of screen height when fully opened)
          child: Container(
            color: Colors.white,
            child: Card(
              margin: EdgeInsets.zero, // Remove margin to fill the entire panel
              elevation: 8, // Increased elevation for better shadow
              shape: const RoundedRectangleBorder(
                // Only round the top corners
                borderRadius: BorderRadius.only(
                  topLeft: Radius.circular(16), // Increased corner radius
                  topRight: Radius.circular(16), // Increased corner radius
                ),
              ),
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize:
                      MainAxisSize.min, // Use min size to avoid overflow
                  children: [
                    // Header row with title and close button
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Flexible(
                          child: Text(
                            'Learning Mode',
                            style: Theme.of(context).textTheme.titleLarge
                                ?.copyWith(fontWeight: FontWeight.bold),
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        IconButton(
                          icon: const Icon(Icons.close),
                          onPressed: onClose,
                          padding: EdgeInsets.zero,
                          constraints: const BoxConstraints(),
                        ),
                      ],
                    ),
                    const Divider(),
                    // Use Expanded with a SingleChildScrollView to handle content overflow
                    Expanded(
                      child: LayoutBuilder(
                        builder: (context, constraints) {
                          return SingleChildScrollView(
                            child: ConstrainedBox(
                              constraints: BoxConstraints(
                                minHeight: constraints.maxHeight,
                              ),
                              child: IntrinsicHeight(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    // Specimen details in a horizontal layout
                                    Row(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        // Left column for basic info
                                        Expanded(
                                          flex: 1,
                                          child: Column(
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            children: [
                                              Text(
                                                specimen['name'] ??
                                                    'Unknown Specimen',
                                                style:
                                                    Theme.of(
                                                      context,
                                                    ).textTheme.titleMedium,
                                              ),
                                              const SizedBox(height: 8),
                                              if (specimen['collection_date'] !=
                                                  null) ...[
                                                Text(
                                                  'Collection Date',
                                                  style: Theme.of(context)
                                                      .textTheme
                                                      .titleSmall
                                                      ?.copyWith(
                                                        fontWeight:
                                                            FontWeight.bold,
                                                      ),
                                                ),
                                                const SizedBox(height: 4),
                                                Text(
                                                  specimen['collection_date'],
                                                  style:
                                                      Theme.of(
                                                        context,
                                                      ).textTheme.bodyMedium,
                                                ),
                                              ],
                                            ],
                                          ),
                                        ),
                                        // Right column for description
                                        if (specimen['description'] != null)
                                          Expanded(
                                            flex: 2,
                                            child: Column(
                                              crossAxisAlignment:
                                                  CrossAxisAlignment.start,
                                              children: [
                                                Text(
                                                  'Description',
                                                  style: Theme.of(context)
                                                      .textTheme
                                                      .titleSmall
                                                      ?.copyWith(
                                                        fontWeight:
                                                            FontWeight.bold,
                                                      ),
                                                ),
                                                const SizedBox(height: 4),
                                                Text(
                                                  specimen['description'],
                                                  style:
                                                      Theme.of(
                                                        context,
                                                      ).textTheme.bodyMedium,
                                                ),
                                              ],
                                            ),
                                          ),
                                      ],
                                    ),
                                    const SizedBox(height: 16),
                                    // Study notes section
                                    Text(
                                      'Study Notes',
                                      style: Theme.of(
                                        context,
                                      ).textTheme.titleSmall?.copyWith(
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                    const SizedBox(height: 4),
                                    SizedBox(
                                      height:
                                          120, // Increased height from 80 to 120
                                      child: TextField(
                                        maxLines: null, // Allow multiple lines
                                        expands:
                                            true, // Fill the available space
                                        textAlignVertical:
                                            TextAlignVertical.top,
                                        style: const TextStyle(fontSize: 14),
                                        decoration: InputDecoration(
                                          labelText: 'Your notes',
                                          border: OutlineInputBorder(
                                            borderRadius: BorderRadius.circular(
                                              8,
                                            ),
                                          ),
                                          contentPadding: const EdgeInsets.all(
                                            12,
                                          ),
                                          hintText:
                                              'Add your study notes here...',
                                          isDense: true, // Help reduce the size
                                          prefixIcon: const Icon(
                                            Icons.edit_note,
                                          ),
                                          suffixIcon: IconButton(
                                            icon: const Icon(Icons.clear),
                                            onPressed: () {
                                              // Clear text functionality would go here
                                              // This would need a controller in a real implementation
                                            },
                                            tooltip: 'Clear notes',
                                          ),
                                        ),
                                      ),
                                    ),
                                    const SizedBox(height: 16),
                                    // Additional features section
                                    Row(
                                      children: [
                                        Expanded(
                                          child: OutlinedButton.icon(
                                            onPressed: () {
                                              // Add image functionality
                                              ScaffoldMessenger.of(
                                                context,
                                              ).showSnackBar(
                                                const SnackBar(
                                                  content: Text(
                                                    'Add image feature coming soon',
                                                  ),
                                                ),
                                              );
                                            },
                                            icon: const Icon(
                                              Icons.image,
                                              size: 18,
                                            ),
                                            label: const Text('Add Image'),
                                            style: OutlinedButton.styleFrom(
                                              padding:
                                                  const EdgeInsets.symmetric(
                                                    vertical: 8,
                                                  ),
                                            ),
                                          ),
                                        ),
                                        const SizedBox(width: 8),
                                        Expanded(
                                          child: OutlinedButton.icon(
                                            onPressed: () {
                                              // Add voice note functionality
                                              ScaffoldMessenger.of(
                                                context,
                                              ).showSnackBar(
                                                const SnackBar(
                                                  content: Text(
                                                    'Voice notes feature coming soon',
                                                  ),
                                                ),
                                              );
                                            },
                                            icon: const Icon(
                                              Icons.mic,
                                              size: 18,
                                            ),
                                            label: const Text('Voice Note'),
                                            style: OutlinedButton.styleFrom(
                                              padding:
                                                  const EdgeInsets.symmetric(
                                                    vertical: 8,
                                                  ),
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                    const SizedBox(height: 16),
                                    Row(
                                      mainAxisAlignment: MainAxisAlignment.end,
                                      children: [
                                        ElevatedButton.icon(
                                          onPressed: () {
                                            // Save notes functionality would go here
                                            ScaffoldMessenger.of(
                                              context,
                                            ).showSnackBar(
                                              const SnackBar(
                                                content: Text('Notes saved'),
                                              ),
                                            );
                                          },
                                          icon: const Icon(
                                            Icons.save,
                                            size: 18,
                                          ),
                                          label: const Text('Save Notes'),
                                          style: ElevatedButton.styleFrom(
                                            padding: const EdgeInsets.symmetric(
                                              horizontal: 12,
                                              vertical: 8,
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

// Specimen info widget
class SpecimenInfoWidget extends StatelessWidget {
  final Map<String, dynamic> specimen;
  final bool isVisible;

  const SpecimenInfoWidget({
    super.key,
    required this.specimen,
    required this.isVisible,
  });

  @override
  Widget build(BuildContext context) {
    if (!isVisible) return const SizedBox.shrink();

    return Positioned(
      left: 10,
      top: 10,
      child: Card(
        elevation: 4,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        child: Padding(
          padding: const EdgeInsets.all(8.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                specimen['name'] ?? 'Unknown Specimen',
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
              if (specimen['description'] != null)
                Text(
                  specimen['description'],
                  style: const TextStyle(fontSize: 12),
                ),
              if (specimen['collection_date'] != null)
                Text(
                  'Collected: ${specimen['collection_date']}',
                  style: const TextStyle(
                    fontSize: 12,
                    fontStyle: FontStyle.italic,
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

// Canvas toolbar widget - modified to fix overflow
class CanvasToolbarWidget extends StatelessWidget {
  final VoidCallback onZoomIn;
  final VoidCallback onZoomOut;
  final VoidCallback onResetZoom;
  final VoidCallback onSave;
  final VoidCallback onShare;

  const CanvasToolbarWidget({
    super.key,
    required this.onZoomIn,
    required this.onZoomOut,
    required this.onResetZoom,
    required this.onSave,
    required this.onShare,
  });

  @override
  Widget build(BuildContext context) {
    return BottomAppBar(
      elevation: 8,
      child: SizedBox(
        height: 50, // Reduced height to avoid overflow
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            IconButton(
              icon: const Icon(Icons.zoom_in),
              onPressed: onZoomIn,
              tooltip: 'Zoom In',
            ),
            IconButton(
              icon: const Icon(Icons.zoom_out),
              onPressed: onZoomOut,
              tooltip: 'Zoom Out',
            ),
            IconButton(
              icon: const Icon(Icons.center_focus_strong),
              onPressed: onResetZoom,
              tooltip: 'Reset',
            ),
            IconButton(
              icon: const Icon(Icons.save),
              onPressed: onSave,
              tooltip: 'Save',
            ),
            IconButton(
              icon: const Icon(Icons.share),
              onPressed: onShare,
              tooltip: 'Share',
            ),
          ],
        ),
      ),
    );
  }
}

// Add this class to your components.dart file
class DrawingPainter extends CustomPainter {
  final List<List<DrawingPoint>> strokes;
  final List<DrawingPoint> currentStroke;
  final ui.Image? image;
  final ui.Image? segmentationImage;
  final bool showSegmentation;

  DrawingPainter({
    required this.strokes,
    required this.currentStroke,
    this.image,
    this.segmentationImage,
    this.showSegmentation = false,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Draw the appropriate image based on the showSegmentation flag
    final ui.Image? imageToDraw = showSegmentation ? segmentationImage : image;
    if (imageToDraw != null) {
      _drawImage(canvas, size, imageToDraw);
    }

    // Draw all the existing strokes
    for (final stroke in strokes) {
      _drawStroke(canvas, stroke);
    }

    // Draw the current stroke being drawn
    if (currentStroke.isNotEmpty) {
      _drawStroke(canvas, currentStroke);
    }
  }

  void _drawImage(Canvas canvas, Size size, ui.Image image) {
    final paint = Paint();
    final src = Rect.fromLTWH(
      0,
      0,
      image.width.toDouble(),
      image.height.toDouble(),
    );
    final dst = Rect.fromLTWH(0, 0, size.width, size.height);
    canvas.drawImageRect(image, src, dst, paint);
  }

  void _drawStroke(Canvas canvas, List<DrawingPoint> stroke) {
    if (stroke.isEmpty) return;

    final path = Path();
    path.moveTo(stroke.first.offset.dx, stroke.first.offset.dy);

    for (int i = 1; i < stroke.length; i++) {
      path.lineTo(stroke[i].offset.dx, stroke[i].offset.dy);
    }

    final paint =
        Paint()
          ..color = stroke.first.color
          ..strokeWidth = stroke.first.strokeWidth
          ..strokeCap = StrokeCap.round
          ..strokeJoin = StrokeJoin.round
          ..style = PaintingStyle.stroke;

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(DrawingPainter oldDelegate) {
    return oldDelegate.strokes != strokes ||
        oldDelegate.currentStroke != currentStroke ||
        oldDelegate.image != image ||
        oldDelegate.segmentationImage != segmentationImage ||
        oldDelegate.showSegmentation != showSegmentation;
  }
}
