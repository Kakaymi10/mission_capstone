import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'dart:typed_data';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'annotation_screen.dart';

class CreateSpecimenDialog extends StatefulWidget {
  final Function(String name, String species, bool isPublic, String? imageUrl)
  onSubmit;

  const CreateSpecimenDialog({Key? key, required this.onSubmit})
    : super(key: key);

  @override
  State<CreateSpecimenDialog> createState() => _CreateSpecimenDialogState();
}

class _CreateSpecimenDialogState extends State<CreateSpecimenDialog> {
  final _formKey = GlobalKey<FormState>();
  final _nameController = TextEditingController();
  String _selectedSpecies = 'plant cells';
  bool _isPublic = true;

  // For cross-platform image handling
  dynamic _imageFile; // Can be File on mobile or Uint8List on web
  Uint8List? _imageBytes; // For displaying images on web
  String? _imagePath; // For tracking the file path (mobile only)

  bool _isLoading = false;
  bool _isAnnotated = false;
  final SupabaseClient _supabase = Supabase.instance.client;
  final ImagePicker _picker = ImagePicker();

  // Get the appropriate API URL based on platform
  String get _apiBaseUrl {
    if (kIsWeb) {
      return 'http://127.0.0.1:8000'; // Web URL
    } else {
      return 'http://10.0.2.2:8000'; // Mobile URL (Android emulator)
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      XFile? pickedFile = await _picker.pickImage(
        source: source,
        maxWidth: 1200,
        maxHeight: 1200,
        imageQuality: 85,
      );

      if (pickedFile == null) return;

      setState(() => _isLoading = true);

      if (kIsWeb) {
        // Handle web platform
        final bytes = await pickedFile.readAsBytes();
        final enhancedBytes = await _enhanceImageWeb(bytes);

        setState(() {
          _imageBytes = enhancedBytes;
          _imageFile = enhancedBytes;
          _isAnnotated = false;
          _isLoading = false;
        });
      } else {
        // Handle mobile platform
        final file = File(pickedFile.path);
        final enhancedFile = await _enhanceImageMobile(file);

        setState(() {
          _imageFile = enhancedFile;
          _imagePath = enhancedFile.path;
          _isAnnotated = false;
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() => _isLoading = false);
      _showErrorSnackBar('Failed to pick image: ${e.toString()}');
    }
  }

  void _showErrorSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red),
    );
  }

  void _showLoadingDialog(String message) {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext context) {
        return AlertDialog(
          content: Row(
            children: [
              const CircularProgressIndicator(),
              const SizedBox(width: 16),
              Text(message),
            ],
          ),
        );
      },
    );
  }

  // Web version of enhance image
  Future<Uint8List> _enhanceImageWeb(Uint8List imageBytes) async {
    try {
      _showLoadingDialog("Enhancing image...");

      // Create multipart request for the API endpoint with web URL
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$_apiBaseUrl/upscale'),
      );

      // Add the image data to the request
      request.files.add(
        http.MultipartFile.fromBytes('file', imageBytes, filename: 'image.jpg'),
      );

      // Send the request
      final response = await request.send();

      if (response.statusCode != 200) {
        if (mounted) Navigator.of(context, rootNavigator: true).pop();
        throw Exception('Failed to enhance image: ${response.statusCode}');
      }

      // Get the response data
      final responseData = await response.stream.toBytes();

      // Close loading dialog
      if (mounted) Navigator.of(context, rootNavigator: true).pop();

      return responseData;
    } catch (e) {
      if (mounted) Navigator.of(context, rootNavigator: true).pop();
      _showErrorSnackBar('Failed to enhance image: ${e.toString()}');
      // Return original image if enhancement fails
      return imageBytes;
    }
  }

  // Mobile version of enhance image
  Future<File> _enhanceImageMobile(File imageFile) async {
    try {
      _showLoadingDialog("Enhancing image...");

      // Create multipart request for API call with mobile URL
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$_apiBaseUrl/upscale'),
      );

      // Add the image file
      request.files.add(
        await http.MultipartFile.fromPath('file', imageFile.path),
      );

      // Send the request
      final response = await request.send();

      if (response.statusCode != 200) {
        if (mounted) Navigator.of(context, rootNavigator: true).pop();
        throw Exception('Failed to enhance image: ${response.statusCode}');
      }

      final responseData = await response.stream.toBytes();

      // Close loading dialog
      if (mounted) Navigator.of(context, rootNavigator: true).pop();

      // Save to a temporary file
      final enhancedImageFile = File('${imageFile.path}_enhanced.png');
      await enhancedImageFile.writeAsBytes(responseData);

      return enhancedImageFile;
    } catch (e) {
      if (mounted) Navigator.of(context, rootNavigator: true).pop();
      _showErrorSnackBar('Failed to enhance image: ${e.toString()}');
      return imageFile; // Return original if enhancement fails
    }
  }

  Future<String?> _uploadImage() async {
    if (_imageFile == null) return null;

    try {
      final String filePath =
          'specimens/${DateTime.now().millisecondsSinceEpoch}_${_nameController.text.toLowerCase().replaceAll(' ', '_')}.png';

      // Handle upload differently based on platform
      if (kIsWeb) {
        // For web, upload bytes
        await _supabase.storage
            .from('specimens')
            .uploadBinary(
              filePath,
              _imageFile as Uint8List,
              fileOptions: const FileOptions(
                cacheControl: '3600',
                upsert: false,
              ),
            );
      } else {
        // For mobile, upload file
        await _supabase.storage
            .from('specimens')
            .upload(
              filePath,
              _imageFile as File,
              fileOptions: const FileOptions(
                cacheControl: '3600',
                upsert: false,
              ),
            );
      }

      // Get the public URL
      final String imageUrl = _supabase.storage
          .from('specimens')
          .getPublicUrl(filePath);
      return imageUrl;
    } on StorageException catch (e) {
      throw Exception('Storage error: ${e.message}');
    } catch (e) {
      throw Exception('Failed to upload image: ${e.toString()}');
    }
  }

  // Add this property to store the annotation data
  Map<String, dynamic>? _annotationData;

  void _navigateToAnnotation() {
    if (_imageFile == null) {
      _showErrorSnackBar('Please select an image first');
      return;
    }

    Navigator.of(context).push(
      MaterialPageRoute(
        builder:
            (context) => AnnotationScreen(
              imageFile: _imageFile,
              onComplete: (result) {
                // This callback will be called by the annotation screen
                if (mounted && result is Map && result['success'] == true) {
                  setState(() {
                    _isAnnotated = true;
                    _annotationData =
                        result
                            .cast<
                              String,
                              dynamic
                            >(); // Store the annotation data
                  });
                }
              },
            ),
      ),
    );
  }

  // Update the createSpecimen method to save annotation data
  Future<void> _createSpecimen() async {
    if (!_formKey.currentState!.validate()) return;

    final user = _supabase.auth.currentUser;
    if (user == null) {
      _showErrorSnackBar('You must be logged in to create a specimen');
      return;
    }

    setState(() => _isLoading = true);
    _showLoadingDialog("Creating specimen...");

    try {
      // Upload image if selected
      String? imageUrl;
      if (_imageFile != null) {
        imageUrl = await _uploadImage();
      }

      // Insert specimen data
      await _supabase.from('specimens').insert({
        'user_id': user.id,
        'name': _nameController.text.trim(),
        'species': _selectedSpecies,
        'is_public': _isPublic,
        'image_url': imageUrl,
        'created_at': DateTime.now().toIso8601String(),
      });

      // Save annotation data if available
      if (_isAnnotated && _annotationData != null) {
        // Create the annotation record with proper references
        final annotationData = {
          'id': _annotationData!['id'],
          'user_id': user.id,
          'annotations': jsonEncode(_annotationData!['annotations']),
          'raw_annotations': jsonEncode(_annotationData!['raw_annotations']),
          'image_width': _annotationData!['image_width'],
          'image_height': _annotationData!['image_height'],
          'image_url': imageUrl, // Link to the specimen image
          'created_at': DateTime.now().toIso8601String(),
        };

        // Save to the image_annotations table
        await _supabase.from('image_annotations').upsert(annotationData);
      }

      if (mounted) {
        Navigator.of(
          context,
          rootNavigator: true,
        ).pop(); // Close loading dialog
        Navigator.of(context).pop(); // Close create specimen dialog

        widget.onSubmit(
          _nameController.text.trim(),
          _selectedSpecies,
          _isPublic,
          imageUrl,
        );

        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Specimen created successfully!'),
            backgroundColor: Colors.green,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        Navigator.of(
          context,
          rootNavigator: true,
        ).pop(); // Close loading dialog
        _showErrorSnackBar('Failed to create specimen: ${e.toString()}');
      }
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Create New Specimen'),
      content: Form(
        key: _formKey,
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildImagePreview(),
              const SizedBox(height: 8),
              _buildImageButtonsRow(),
              const SizedBox(height: 16),
              _buildNameField(),
              const SizedBox(height: 16),
              _buildSpeciesDropdown(),
              const SizedBox(height: 16),
              _buildVisibilitySwitch(),
            ],
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: _isLoading ? null : () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        ElevatedButton(
          onPressed: _isLoading ? null : _createSpecimen,
          child:
              _isLoading
                  ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                  : const Text('Create'),
        ),
      ],
    );
  }

  Widget _buildImagePreview() {
    return Stack(
      children: [
        Container(
          height: 150,
          width: double.infinity,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: Colors.grey.shade300),
          ),
          child:
              _hasImage()
                  ? ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: _buildImageWidget(),
                  )
                  : Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: const [
                      Icon(Icons.image, size: 48, color: Colors.grey),
                      SizedBox(height: 8),
                      Text(
                        'No image selected',
                        style: TextStyle(color: Colors.grey),
                      ),
                    ],
                  ),
        ),
        if (_isAnnotated)
          Positioned(
            top: 8,
            right: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.green,
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.edit, color: Colors.white, size: 14),
                  SizedBox(width: 4),
                  Text(
                    'Annotated',
                    style: TextStyle(color: Colors.white, fontSize: 12),
                  ),
                ],
              ),
            ),
          ),
      ],
    );
  }

  bool _hasImage() {
    if (kIsWeb) {
      return _imageBytes != null;
    } else {
      return _imageFile != null;
    }
  }

  Widget _buildImageWidget() {
    if (kIsWeb) {
      if (_imageBytes != null) {
        return Image.memory(_imageBytes!, fit: BoxFit.cover);
      }
    } else {
      if (_imageFile != null) {
        return Image.file(_imageFile as File, fit: BoxFit.cover);
      }
    }

    // Fallback
    return const SizedBox();
  }

  Widget _buildImageButtonsRow() {
    return Wrap(
      alignment: WrapAlignment.center,
      spacing: 8,
      children: [
        ElevatedButton.icon(
          icon: const Icon(Icons.photo_library),
          label: const Text('Gallery'),
          onPressed: _isLoading ? null : () => _pickImage(ImageSource.gallery),
        ),
        ElevatedButton.icon(
          icon: const Icon(Icons.camera_alt),
          label: const Text('Camera'),
          onPressed: _isLoading ? null : () => _pickImage(ImageSource.camera),
        ),
        if (_hasImage())
          ElevatedButton.icon(
            icon: const Icon(Icons.edit),
            label: const Text('Annotate'),
            onPressed: _isLoading ? null : _navigateToAnnotation,
            style: ElevatedButton.styleFrom(
              backgroundColor: _isAnnotated ? Colors.green.shade700 : null,
            ),
          ),
      ],
    );
  }

  Widget _buildNameField() {
    return TextFormField(
      controller: _nameController,
      decoration: const InputDecoration(
        labelText: 'Specimen Name',
        border: OutlineInputBorder(),
      ),
      validator: (value) {
        if (value == null || value.trim().isEmpty) {
          return 'Please enter a specimen name';
        }
        return null;
      },
      enabled: !_isLoading,
    );
  }

  Widget _buildSpeciesDropdown() {
    return DropdownButtonFormField<String>(
      value: _selectedSpecies,
      decoration: const InputDecoration(
        labelText: 'Species Type',
        border: OutlineInputBorder(),
      ),
      items: const [
        DropdownMenuItem(value: 'plant cells', child: Text('Plant Cells')),
        DropdownMenuItem(value: 'animal cells', child: Text('Animal Cells')),
        DropdownMenuItem(value: 'bacteria', child: Text('Bacteria')),
        DropdownMenuItem(value: 'fungi', child: Text('Fungi')),
        DropdownMenuItem(
          value: 'tissue samples',
          child: Text('Tissue Samples'),
        ),
      ],
      onChanged:
          _isLoading
              ? null
              : (value) => setState(() => _selectedSpecies = value!),
    );
  }

  Widget _buildVisibilitySwitch() {
    return SwitchListTile(
      title: const Text('Make Public'),
      subtitle: const Text('Allow others to view this specimen'),
      value: _isPublic,
      onChanged:
          _isLoading ? null : (value) => setState(() => _isPublic = value),
    );
  }

  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }
}
