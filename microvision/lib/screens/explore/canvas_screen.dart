// ArcaneSpecimenViewer with magical segmentation and enchanted chat
import 'dart:convert';
import 'dart:math';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'dart:ui' as ui;
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:image_gallery_saver/image_gallery_saver.dart';
import 'package:microvision/screens/explore/quizz_screen.dart';

class ArcaneSpecimenViewer extends StatefulWidget {
  final Map<String, dynamic> magicalSpecimen;

  const ArcaneSpecimenViewer({super.key, required this.magicalSpecimen});

  @override
  _ArcaneSpecimenViewerState createState() => _ArcaneSpecimenViewerState();
}

class _ArcaneSpecimenViewerState extends State<ArcaneSpecimenViewer>
    with SingleTickerProviderStateMixin {
  // Mystical image and arcane segmentation state
  late Size _runestone;
  bool _isRunestoneEmpowered = false;
  ui.Image? _specimenRelic;
  bool _isConjuringMagicVeil = false;
  ui.Image? _magicVeil;
  bool _showMagicVeil = false;
  Offset? _lastEtherealTouchpoint;
  bool _isArcaneRevealMode = false;

  // Magical particle effects
  late AnimationController _magicController;
  final List<MagicalParticle> _magicParticles = [];

  // Enchanted scroll state
  bool _showEnchantedScroll = false;
  bool _isChannelingWisdom = false;
  final TextEditingController _scrollController = TextEditingController();
  final List<Map<String, String>> _arcaneConversation = [];
  String? _arcaneProphecy;

  final GlobalKey _runicKey = GlobalKey();

  @override
  void initState() {
    super.initState();
    _invokeRunestone();

    _magicController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 3000),
    )..repeat();
    _magicController.addListener(_animateMagicParticles);
  }

  @override
  void dispose() {
    _magicController.dispose();
    super.dispose();
  }

  void _animateMagicParticles() {
    setState(() {
      // Update existing particles
      for (int i = _magicParticles.length - 1; i >= 0; i--) {
        _magicParticles[i].update();
        if (_magicParticles[i].lifespan <= 0) {
          _magicParticles.removeAt(i);
        }
      }

      // Add new particles occasionally
      if (_showMagicVeil &&
          _magicParticles.length < 50 &&
          _lastEtherealTouchpoint != null) {
        if (DateTime.now().millisecondsSinceEpoch % 10 == 0) {
          _magicParticles.add(
            MagicalParticle(position: _lastEtherealTouchpoint!),
          );
        }
      }
    });
  }

  Future<void> _invokeRunestone() async {
    if (widget.magicalSpecimen['image_url'] == null) return;

    final ImageProvider provider = NetworkImage(
      widget.magicalSpecimen['image_url'],
    );
    final ImageStream stream = provider.resolve(const ImageConfiguration());

    stream.addListener(
      ImageStreamListener((ImageInfo info, bool _) {
        setState(() {
          _runestone = Size(
            info.image.width.toDouble(),
            info.image.height.toDouble(),
          );
          _isRunestoneEmpowered = true;
          _specimenRelic = info.image;
        });
      }),
    );
  }

  Offset _translateToRuneCoordinates(Offset screenPoint) {
    final RenderBox renderBox =
        _runicKey.currentContext!.findRenderObject() as RenderBox;
    final Offset localPosition = renderBox.globalToLocal(screenPoint);

    final double scaleX = _runestone.width / renderBox.size.width;
    final double scaleY = _runestone.height / renderBox.size.height;

    return Offset(localPosition.dx * scaleX, localPosition.dy * scaleY);
  }

  Future<void> _castArcaneRevealment(Offset touchPoint) async {
    if (!_isRunestoneEmpowered) return;

    setState(() {
      _isConjuringMagicVeil = true;
      _lastEtherealTouchpoint = touchPoint;

      // Add burst of particles at touch point
      for (int i = 0; i < 20; i++) {
        _magicParticles.add(MagicalParticle(position: touchPoint));
      }
    });

    try {
      final response = await http.get(
        Uri.parse(widget.magicalSpecimen['image_url']),
      );
      if (response.statusCode != 200) return;

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('http://127.0.0.1:8000/segment_click/'),
      );

      request.files.add(
        http.MultipartFile.fromBytes(
          'image',
          response.bodyBytes,
          filename: 'arcane_specimen.jpg',
        ),
      );

      request.fields['point_x'] = touchPoint.dx.toStringAsFixed(2);
      request.fields['point_y'] = touchPoint.dy.toStringAsFixed(2);
      request.fields['point_type'] = '1';
      request.fields['conf'] = '0.4';
      request.fields['iou'] = '0.9';

      final streamedResponse = await request.send();
      final segmentationResponse = await http.Response.fromStream(
        streamedResponse,
      );

      if (segmentationResponse.statusCode != 200) return;

      final Uint8List enchantedImageBytes = segmentationResponse.bodyBytes;
      final codec = await ui.instantiateImageCodec(enchantedImageBytes);
      final frameInfo = await codec.getNextFrame();

      setState(() {
        _magicVeil = frameInfo.image;
        _showMagicVeil = true;
        _isConjuringMagicVeil = false;
      });
    } catch (e) {
      setState(() => _isConjuringMagicVeil = false);
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Arcane error: ${e.toString()}')));
    }
  }

  Future<void> _channelArcaneProphecy() async {
    if (_magicVeil == null) return;

    setState(() => _isChannelingWisdom = true);

    try {
      final ByteData? byteData = await _magicVeil!.toByteData();
      if (byteData == null) return;

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('http://127.0.0.1:8000/summarize/'),
      );

      request.files.add(
        http.MultipartFile.fromBytes(
          'image',
          byteData.buffer.asUint8List(),
          filename: 'arcane_veil.png',
        ),
      );

      final response = await request.send();
      final responseString = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        setState(() {
          _arcaneProphecy = responseString;
          _isChannelingWisdom = false;
        });
      }
    } catch (e) {
      setState(() => _isChannelingWisdom = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Divination error: ${e.toString()}')),
      );
    }
  }

  Future<void> _sendArcaneCommunique() async {
    if (_magicVeil == null || _scrollController.text.isEmpty) return;

    final message = _scrollController.text;
    _scrollController.clear();

    setState(() {
      _arcaneConversation.add({'role': 'mage', 'content': message});
      _isChannelingWisdom = true;

      // Add mystical particles around the scroll
      for (int i = 0; i < 30; i++) {
        _magicParticles.add(
          MagicalParticle(
            position: Offset(
              300 * math.Random().nextDouble(),
              400 * math.Random().nextDouble(),
            ),
            color: Colors.purpleAccent,
          ),
        );
      }
    });

    try {
      // Get the byte data of the segmented image
      final ByteData? byteData = await _magicVeil!.toByteData(
        format: ui.ImageByteFormat.png,
      );
      if (byteData == null) return;

      // Calculate the bounding box of the segmented region
      final region = await _getEnchantedRegionBounds();
      if (region == null) return;

      // Create the multipart request
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('http://127.0.0.1:8000/chat_region/'),
      );

      // Add the image file
      request.files.add(
        http.MultipartFile.fromBytes(
          'image',
          byteData.buffer.asUint8List(),
          filename: 'arcane_region.png',
        ),
      );

      // Add the required fields
      request.fields['question'] = message;
      request.fields['bbox'] =
          '${region.left.toInt()},${region.top.toInt()},'
          '${region.right.toInt()},${region.bottom.toInt()}';

      // Add optional label if available
      if (widget.magicalSpecimen['name'] != null) {
        request.fields['label'] = widget.magicalSpecimen['name'];
      }

      // Send the request
      final response = await request.send();

      if (response.statusCode == 200) {
        final responseString = await response.stream.bytesToString();
        final responseJson = jsonDecode(responseString);

        setState(() {
          _arcaneConversation.add({
            'role': 'familiar',
            'content': responseJson['answer'] ?? 'The spirits are silent...',
          });
          _isChannelingWisdom = false;
        });
      } else {
        throw Exception(
          'The arcane spirits failed to respond: ${response.statusCode}',
        );
      }
    } catch (e) {
      setState(() => _isChannelingWisdom = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Communion error: ${e.toString()}')),
      );
    }
  }

  Future<Rect?> _getEnchantedRegionBounds() async {
    if (_magicVeil == null) return null;

    final byteData = await _magicVeil!.toByteData();
    if (byteData == null) return null;

    final magicEssence = byteData.buffer.asUint8List();
    final width = _magicVeil!.width;
    final height = _magicVeil!.height;

    int minX = width;
    int minY = height;
    int maxX = 0;
    int maxY = 0;

    // Find the bounds of magical aura (non-transparent pixels)
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final index = (y * width + x) * 4;
        if (magicEssence[index + 3] > 0) {
          // Check ethereal presence (alpha channel)
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    // Add mystical margin and bind to dimensional boundaries
    const margin = 10;
    minX = (minX - margin).clamp(0, width);
    minY = (minY - margin).clamp(0, height);
    maxX = (maxX + margin).clamp(0, width);
    maxY = (maxY + margin).clamp(0, height);

    return Rect.fromLTRB(
      minX.toDouble(),
      minY.toDouble(),
      maxX.toDouble(),
      maxY.toDouble(),
    );
  }

  void _toggleEnchantedScroll() {
    setState(() => _showEnchantedScroll = !_showEnchantedScroll);
  }

  void _navigateToQuizScreen(BuildContext context) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder:
            (context) => ArcaneQuizScreen(
              specimenName:
                  widget.magicalSpecimen['name'] ?? 'Magical Specimen',
              specimenImageUrl: widget.magicalSpecimen['image_url'],
            ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.magicalSpecimen['name'] ?? 'Magical Specimen'),
        backgroundColor: Colors.indigo.shade900,
        foregroundColor: Colors.amber,
        actions: [
          IconButton(
            icon: Icon(Icons.quiz, color: Colors.amber),
            onPressed: () => _navigateToQuizScreen(context),
            tooltip: 'Take the Arcane Quiz',
          ),

          IconButton(
            icon: Icon(
              _isArcaneRevealMode ? Icons.auto_fix_high : Icons.auto_fix_normal,
              color:
                  _isArcaneRevealMode
                      ? Colors.amber
                      : Colors.amber.withOpacity(0.7),
            ),
            onPressed:
                () => setState(() {
                  _isArcaneRevealMode = !_isArcaneRevealMode;
                }),
            tooltip: 'Toggle Arcane Reveal Mode',
          ),
          IconButton(
            icon: Icon(
              _showMagicVeil ? Icons.remove_red_eye : Icons.visibility_off,
              color:
                  _showMagicVeil ? Colors.amber : Colors.amber.withOpacity(0.7),
            ),
            onPressed:
                _magicVeil != null
                    ? () => setState(() => _showMagicVeil = !_showMagicVeil)
                    : null,
            tooltip: 'Toggle Arcane Veil',
          ),
          IconButton(
            icon: Icon(Icons.menu_book, color: Colors.amber),
            onPressed: _magicVeil != null ? _toggleEnchantedScroll : null,
            tooltip: 'Open Enchanted Scroll',
          ),
        ],
      ),
      body:
          _isRunestoneEmpowered
              ? Stack(
                children: [
                  // Background magic gradient
                  Container(
                    decoration: BoxDecoration(
                      gradient: RadialGradient(
                        colors: [Colors.indigo.shade100, Colors.black],
                        center: Alignment.center,
                        radius: 1.5,
                      ),
                    ),
                  ),

                  RepaintBoundary(
                    key: _runicKey,
                    child: GestureDetector(
                      onTapDown: (details) {
                        if (_isArcaneRevealMode) {
                          _castArcaneRevealment(
                            _translateToRuneCoordinates(details.globalPosition),
                          );
                        }
                      },
                      child: CustomPaint(
                        size: _runestone,
                        painter: _ArcanePainter(
                          specimenRelic: _specimenRelic,
                          magicVeil: _showMagicVeil ? _magicVeil : null,
                          etherTouch: _lastEtherealTouchpoint,
                          magicParticles: _magicParticles,
                        ),
                        foregroundPainter:
                            _isArcaneRevealMode ? _MysticGlowPainter() : null,
                      ),
                    ),
                  ),
                  if (_isConjuringMagicVeil)
                    Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          SizedBox(
                            width: 60,
                            height: 60,
                            child: CircularProgressIndicator(
                              color: Colors.purpleAccent,
                              backgroundColor: const Color.fromARGB(
                                255,
                                41,
                                57,
                                231,
                              ),
                              strokeWidth: 6,
                            ),
                          ),
                          const SizedBox(height: 16),
                          Text(
                            "Conjuring Arcane Energies...",
                            style: TextStyle(
                              color: Colors.amber,
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                              shadows: [
                                Shadow(
                                  color: Colors.purple,
                                  blurRadius: 10,
                                  offset: Offset(0, 0),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  if (_showEnchantedScroll) _buildEnchantedScroll(),
                ],
              )
              : Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    SizedBox(
                      width: 80,
                      height: 80,
                      child: CircularProgressIndicator(
                        color: Colors.amber,
                        backgroundColor: Colors.indigo.shade900,
                        strokeWidth: 8,
                      ),
                    ),
                    const SizedBox(height: 24),
                    Text(
                      "Empowering Magical Runestone...",
                      style: TextStyle(
                        color: Colors.amber,
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        shadows: [
                          Shadow(
                            color: Colors.purple,
                            blurRadius: 10,
                            offset: Offset(0, 0),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
      floatingActionButton:
          _isRunestoneEmpowered
              ? FloatingActionButton(
                onPressed: () {
                  // Capture current view to gallery
                  // Implementation would go here
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text(
                        'Magical moment preserved in your grimoire',
                      ),
                    ),
                  );
                },
                backgroundColor: Colors.amber,
                child: Icon(Icons.camera_alt, color: Colors.indigo.shade900),
              )
              : null,
    );
  }

  Widget _buildEnchantedScroll() {
    return Align(
      alignment: Alignment.bottomRight,
      child: Container(
        width: 300,
        height: 400,
        margin: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.brown.shade200,
          borderRadius: BorderRadius.circular(8),
          boxShadow: [
            BoxShadow(
              color: Colors.purple.withOpacity(0.3),
              blurRadius: 15,
              spreadRadius: 5,
              offset: const Offset(0, 0),
            ),
          ],
          image: DecorationImage(
            image: AssetImage('assets/parchment_texture.png'),
            fit: BoxFit.cover,
          ),
        ),
        child: Stack(
          children: [
            // Magical particle effect overlay
            CustomPaint(
              size: Size(300, 400),
              painter: _ScrollParticlePainter(particles: _magicParticles),
            ),

            Column(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.brown.shade400,
                    borderRadius: BorderRadius.only(
                      topLeft: Radius.circular(8),
                      topRight: Radius.circular(8),
                    ),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.menu_book, color: Colors.amber),
                      const SizedBox(width: 8),
                      Text(
                        "Enchanted Scroll",
                        style: TextStyle(
                          color: Colors.amber,
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
                Expanded(
                  child: ListView.builder(
                    padding: const EdgeInsets.all(8),
                    itemCount: _arcaneConversation.length,
                    itemBuilder: (context, index) {
                      final message = _arcaneConversation[index];
                      return Container(
                        margin: const EdgeInsets.symmetric(vertical: 4),
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color:
                              message['role'] == 'mage'
                                  ? Colors.indigo.withOpacity(0.2)
                                  : Colors.amber.withOpacity(0.2),
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(
                            color:
                                message['role'] == 'mage'
                                    ? Colors.indigo.withOpacity(0.5)
                                    : Colors.amber.withOpacity(0.5),
                            width: 1,
                          ),
                        ),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Icon(
                              message['role'] == 'mage'
                                  ? Icons.person
                                  : Icons.auto_awesome,
                              color:
                                  message['role'] == 'mage'
                                      ? Colors.indigo
                                      : Colors.amber.shade700,
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                message['content'] ?? '',
                                style: TextStyle(
                                  color: Colors.black87,
                                  fontStyle:
                                      message['role'] == 'familiar'
                                          ? FontStyle.italic
                                          : FontStyle.normal,
                                ),
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ),
                if (_isChannelingWisdom)
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 8),
                    child: LinearProgressIndicator(
                      color: Colors.amber,
                      backgroundColor: Colors.indigo.withOpacity(0.3),
                    ),
                  ),
                Padding(
                  padding: const EdgeInsets.all(8),
                  child: Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: _scrollController,
                          decoration: InputDecoration(
                            hintText: 'Ask the arcane familiar...',
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(16),
                              borderSide: BorderSide(color: Colors.indigo),
                            ),
                            filled: true,
                            fillColor: Colors.white.withOpacity(0.8),
                          ),
                          onSubmitted: (_) => _sendArcaneCommunique(),
                        ),
                      ),
                      IconButton(
                        icon: Icon(Icons.send, color: Colors.indigo),
                        onPressed: _sendArcaneCommunique,
                      ),
                    ],
                  ),
                ),
                if (_arcaneProphecy != null)
                  Padding(
                    padding: const EdgeInsets.all(8),
                    child: Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: Colors.amber.withOpacity(0.3),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                          color: Colors.amber.shade700,
                          width: 1,
                        ),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Icon(
                                Icons.auto_awesome,
                                color: Colors.amber.shade700,
                              ),
                              const SizedBox(width: 4),
                              Text(
                                'Arcane Prophecy:',
                                style: TextStyle(
                                  fontWeight: FontWeight.bold,
                                  color: Colors.amber.shade900,
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 4),
                          Text(
                            _arcaneProphecy!,
                            style: TextStyle(
                              fontStyle: FontStyle.italic,
                              color: Colors.indigo.shade900,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                TextButton.icon(
                  onPressed: _channelArcaneProphecy,
                  icon: Icon(Icons.auto_awesome, color: Colors.amber.shade800),
                  label: Text(
                    'Channel Arcane Prophecy',
                    style: TextStyle(
                      color: Colors.amber.shade800,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  style: TextButton.styleFrom(
                    backgroundColor: Colors.indigo.withOpacity(0.2),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class MagicalParticle {
  Offset position;
  Offset velocity;
  double size;
  Color color;
  int lifespan;

  MagicalParticle({required this.position, Color? color})
    : velocity = Offset(
        (math.Random().nextDouble() * 2 - 1) * 2,
        (math.Random().nextDouble() * 2 - 1) * 2,
      ),
      size = math.Random().nextDouble() * 6 + 2,
      color = color ?? _getRandomMagicColor(),
      lifespan = math.Random().nextInt(40) + 20;

  static Color _getRandomMagicColor() {
    final colors = [
      Colors.purple.shade300,
      Colors.blue.shade300,
      Colors.cyan.shade300,
      Colors.amber.shade300,
      Colors.deepPurple.shade300,
    ];
    return colors[Random().nextInt(colors.length)];
  }

  void update() {
    position += velocity;
    lifespan--;
    // Add slight random movement
    velocity += Offset(
      (math.Random().nextDouble() * 2 - 1) * 0.1,
      (math.Random().nextDouble() * 2 - 1) * 0.1,
    );
  }
}

class _ArcanePainter extends CustomPainter {
  final ui.Image? specimenRelic;
  final ui.Image? magicVeil;
  final Offset? etherTouch;
  final List<MagicalParticle> magicParticles;

  _ArcanePainter({
    required this.specimenRelic,
    required this.magicVeil,
    required this.etherTouch,
    required this.magicParticles,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (specimenRelic != null) {
      canvas.drawImageRect(
        specimenRelic!,
        Rect.fromLTWH(
          0,
          0,
          specimenRelic!.width.toDouble(),
          specimenRelic!.height.toDouble(),
        ),
        Rect.fromLTWH(0, 0, size.width, size.height),
        Paint(),
      );
    }

    if (magicVeil != null) {
      // Draw magical glow around the segmentation
      final glowPaint =
          Paint()
            ..color = Colors.purple.withOpacity(0.3)
            ..maskFilter = MaskFilter.blur(BlurStyle.normal, 20);

      canvas.drawImageRect(
        magicVeil!,
        Rect.fromLTWH(
          0,
          0,
          magicVeil!.width.toDouble(),
          magicVeil!.height.toDouble(),
        ),
        Rect.fromLTWH(0, 0, size.width, size.height),
        glowPaint,
      );

      // Draw the actual segmentation
      canvas.drawImageRect(
        magicVeil!,
        Rect.fromLTWH(
          0,
          0,
          magicVeil!.width.toDouble(),
          magicVeil!.height.toDouble(),
        ),
        Rect.fromLTWH(0, 0, size.width, size.height),
        Paint(),
      );
    }

    // Draw magical particles
    for (final particle in magicParticles) {
      final particlePaint =
          Paint()
            ..color = particle.color.withOpacity(particle.lifespan / 60)
            ..style = PaintingStyle.fill
            ..maskFilter = MaskFilter.blur(BlurStyle.normal, particle.size / 3);

      canvas.drawCircle(particle.position, particle.size, particlePaint);
    }

    if (etherTouch != null) {
      // Draw ethereal touch point with a glowing effect
      final outerGlowPaint =
          Paint()
            ..color = Colors.amber.withOpacity(0.3)
            ..style = PaintingStyle.fill
            ..maskFilter = MaskFilter.blur(BlurStyle.normal, 10);

      final innerGlowPaint =
          Paint()
            ..color = Colors.amber.withOpacity(0.7)
            ..style = PaintingStyle.fill
            ..maskFilter = MaskFilter.blur(BlurStyle.normal, 5);

      final corePaint =
          Paint()
            ..color = Colors.amber
            ..style = PaintingStyle.fill;

      canvas.drawCircle(etherTouch!, 15, outerGlowPaint);
      canvas.drawCircle(etherTouch!, 10, innerGlowPaint);
      canvas.drawCircle(etherTouch!, 5, corePaint);
    }
  }

  @override
  bool shouldRepaint(covariant _ArcanePainter oldDelegate) {
    return oldDelegate.specimenRelic != specimenRelic ||
        oldDelegate.magicVeil != magicVeil ||
        oldDelegate.etherTouch != etherTouch ||
        oldDelegate.magicParticles.length != magicParticles.length;
  }
}

class _MysticGlowPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    // Show a subtle cursor glow to indicate arcane reveal mode is active
    final Paint cursorPaint =
        Paint()
          ..shader = RadialGradient(
            colors: [Colors.amber.withOpacity(0.2), Colors.transparent],
          ).createShader(
            Rect.fromCircle(
              center: Offset(size.width / 2, size.height / 2),
              radius: 40,
            ),
          );

    canvas.drawCircle(Offset(size.width / 2, size.height / 2), 40, cursorPaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return false;
  }
}

class _ScrollParticlePainter extends CustomPainter {
  final List<MagicalParticle> particles;

  _ScrollParticlePainter({required this.particles});

  @override
  void paint(Canvas canvas, Size size) {
    for (final particle in particles) {
      final particlePaint =
          Paint()
            ..color = particle.color.withOpacity(particle.lifespan / 60)
            ..style = PaintingStyle.fill
            ..maskFilter = MaskFilter.blur(BlurStyle.normal, particle.size / 3);

      canvas.drawCircle(
        particle.position,
        particle.size / 2, // Smaller particles for the scroll
        particlePaint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true; // Always repaint for animations
  }
}

// EXTENSIONS FOR MAGICAL EFFECTS

// Extension for magical text animations
class EnchantedText extends StatefulWidget {
  final String text;
  final TextStyle style;
  final Duration duration;

  const EnchantedText({
    Key? key,
    required this.text,
    required this.style,
    this.duration = const Duration(milliseconds: 2000),
  }) : super(key: key);

  @override
  _EnchantedTextState createState() => _EnchantedTextState();
}

class _EnchantedTextState extends State<EnchantedText>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _glowAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: widget.duration)
      ..repeat(reverse: true);

    _glowAnimation = Tween<double>(
      begin: 0,
      end: 5,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Text(
          widget.text,
          style: widget.style.copyWith(
            shadows: [
              Shadow(
                color: Colors.purple.withOpacity(_controller.value * 0.7),
                blurRadius: _glowAnimation.value,
                offset: Offset(0, 0),
              ),
            ],
          ),
        );
      },
    );
  }
}

// Magical Ripple Effect for when spells are cast
class MagicalRippleEffect extends StatefulWidget {
  final Widget child;
  final Color rippleColor;
  final Duration duration;

  const MagicalRippleEffect({
    Key? key,
    required this.child,
    this.rippleColor = Colors.purple,
    this.duration = const Duration(milliseconds: 1500),
  }) : super(key: key);

  @override
  _MagicalRippleEffectState createState() => _MagicalRippleEffectState();
}

class _MagicalRippleEffectState extends State<MagicalRippleEffect>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: widget.duration)
      ..forward();

    _controller.addStatusListener((status) {
      if (status == AnimationStatus.completed) {
        _controller.reset();
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.passthrough,
      children: [
        widget.child,
        AnimatedBuilder(
          animation: _controller,
          builder: (_, __) {
            return CustomPaint(
              size: Size.infinite,
              painter: _RipplePainter(
                color: widget.rippleColor,
                animationValue: _controller.value,
              ),
            );
          },
        ),
      ],
    );
  }
}

class _RipplePainter extends CustomPainter {
  final Color color;
  final double animationValue;

  _RipplePainter({required this.color, required this.animationValue});

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final maxRadius = size.width > size.height ? size.width : size.height;

    final currentRadius = maxRadius * animationValue;

    final paint =
        Paint()
          ..color = color.withOpacity(1.0 - animationValue)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3.0 * (1.0 - animationValue);

    canvas.drawCircle(center, currentRadius, paint);

    // Add some magical sparkles along the ripple
    if (animationValue > 0.1 && animationValue < 0.9) {
      final sparkleCount = 12;
      final sparkleRadius = 3.0 * (1.0 - animationValue);

      for (int i = 0; i < sparkleCount; i++) {
        final angle = 2 * pi * i / sparkleCount;
        final offset = Offset(
          center.dx + currentRadius * cos(angle),
          center.dy + currentRadius * sin(angle),
        );

        final sparklePaint =
            Paint()
              ..color = Colors.white.withOpacity(1.0 - animationValue)
              ..style = PaintingStyle.fill
              ..maskFilter = MaskFilter.blur(BlurStyle.normal, sparkleRadius);

        canvas.drawCircle(offset, sparkleRadius * 2, sparklePaint);
      }
    }
  }

  @override
  bool shouldRepaint(_RipplePainter oldDelegate) {
    return oldDelegate.animationValue != animationValue;
  }
}

// Enchanted Button for magical UI interactions
class EnchantedButton extends StatefulWidget {
  final VoidCallback onPressed;
  final Widget child;
  final Color glowColor;

  const EnchantedButton({
    Key? key,
    required this.onPressed,
    required this.child,
    this.glowColor = Colors.purple,
  }) : super(key: key);

  @override
  _EnchantedButtonState createState() => _EnchantedButtonState();
}

class _EnchantedButtonState extends State<EnchantedButton>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _glowAnimation;
  bool _isHovered = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);

    _glowAnimation = Tween<double>(
      begin: 2.0,
      end: 8.0,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) => setState(() => _isHovered = true),
      onExit: (_) => setState(() => _isHovered = false),
      child: AnimatedBuilder(
        animation: _controller,
        builder: (context, child) {
          return Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(12),
              boxShadow:
                  _isHovered
                      ? [
                        BoxShadow(
                          color: widget.glowColor.withOpacity(0.5),
                          blurRadius: _glowAnimation.value * 2,
                          spreadRadius: _glowAnimation.value / 2,
                        ),
                      ]
                      : [],
            ),
            child: Material(
              color: Colors.transparent,
              child: InkWell(
                onTap: () {
                  // Add ripple effect when pressed
                  widget.onPressed();
                },
                borderRadius: BorderRadius.circular(12),
                splashColor: widget.glowColor.withOpacity(0.3),
                hoverColor: widget.glowColor.withOpacity(0.1),
                child: Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 16.0,
                    vertical: 8.0,
                  ),
                  child: widget.child,
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

// Magic Theme Provider for consistent magical styling
class MagicTheme {
  static ThemeData get darkMagic {
    return ThemeData.dark().copyWith(
      primaryColor: Colors.indigo.shade900,
      scaffoldBackgroundColor: Colors.black,
      colorScheme: ColorScheme.dark(
        primary: Colors.indigo.shade900,
        secondary: Colors.amber,
        surface: Colors.indigo.shade900,
        background: Colors.black,
        error: Colors.red.shade700,
      ),
      textTheme: TextTheme(
        headlineLarge: TextStyle(
          color: Colors.amber,
          fontWeight: FontWeight.bold,
          shadows: [
            Shadow(
              color: Colors.purple.withOpacity(0.5),
              blurRadius: 5,
              offset: Offset(0, 0),
            ),
          ],
        ),
        bodyLarge: TextStyle(color: Colors.grey.shade300),
        bodyMedium: TextStyle(color: Colors.grey.shade300),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.indigo.shade900,
          foregroundColor: Colors.amber,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
      ),
    );
  }
}
