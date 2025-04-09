// 1. models.dart - Contains data models and painters
import 'package:flutter/material.dart';
import 'dart:ui' as ui;

// Drawing point model for the canvas
import 'package:flutter/material.dart';

// Drawing point model
class DrawingPoint {
  final Offset offset;
  final Color color;
  final double strokeWidth;

  DrawingPoint({
    required this.offset,
    required this.color,
    required this.strokeWidth,
  });
}

// Custom painter for annotations
class DrawingPainter extends CustomPainter {
  final List<List<DrawingPoint>> strokes;
  final List<DrawingPoint> currentStroke;
  final ui.Image? image;

  DrawingPainter({
    required this.strokes,
    required this.currentStroke,
    this.image,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Draw all completed strokes
    for (final stroke in strokes) {
      _drawStroke(canvas, stroke);
    }

    // Draw the current stroke being drawn
    if (currentStroke.isNotEmpty) {
      _drawStroke(canvas, currentStroke);
    }
  }

  void _drawStroke(Canvas canvas, List<DrawingPoint> points) {
    if (points.isEmpty) return;

    final path = Path();
    path.moveTo(points[0].offset.dx, points[0].offset.dy);

    if (points.length < 2) {
      // For a single point, draw a dot
      canvas.drawCircle(
        points[0].offset,
        points[0].strokeWidth / 2,
        Paint()
          ..color = points[0].color
          ..strokeWidth = points[0].strokeWidth
          ..strokeCap = StrokeCap.round,
      );
      return;
    }

    // For multiple points, create a smooth path
    for (int i = 1; i < points.length; i++) {
      final p0 = points[i - 1].offset;
      final p1 = points[i].offset;

      if (i + 1 < points.length) {
        final p2 = points[i + 1].offset;

        // Calculate control points for a smooth curve
        final c1 = Offset(
          p0.dx + (p1.dx - p0.dx) * 0.5,
          p0.dy + (p1.dy - p0.dy) * 0.5,
        );
        final c2 = Offset(
          p1.dx + (p2.dx - p1.dx) * 0.5,
          p1.dy + (p2.dy - p1.dy) * 0.5,
        );

        path.cubicTo(c1.dx, c1.dy, c2.dx, c2.dy, p1.dx, p1.dy);
      } else {
        // For the last segment, just use a quadratic curve
        path.quadraticBezierTo(
          p0.dx,
          p0.dy,
          (p0.dx + p1.dx) / 2,
          (p0.dy + p1.dy) / 2,
        );
      }
    }

    canvas.drawPath(
      path,
      Paint()
        ..color = points[0].color
        ..strokeWidth = points[0].strokeWidth
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round
        ..isAntiAlias = true,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    if (oldDelegate is DrawingPainter) {
      return oldDelegate.strokes != strokes ||
          oldDelegate.currentStroke != currentStroke ||
          oldDelegate.image != image;
    }
    return true;
  }
}
