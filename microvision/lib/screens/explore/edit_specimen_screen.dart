import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class EditSpecimenScreen extends StatefulWidget {
  final Map<String, dynamic> specimen;
  final Function? onSpecimenUpdated;

  const EditSpecimenScreen({
    super.key,
    required this.specimen,
    this.onSpecimenUpdated,
  });

  @override
  _EditSpecimenScreenState createState() => _EditSpecimenScreenState();
}

class _EditSpecimenScreenState extends State<EditSpecimenScreen> {
  final _formKey = GlobalKey<FormState>();
  final _nameController = TextEditingController();
  final _speciesController = TextEditingController();
  bool _isPublic = false;
  bool _isLoading = false;
  final _supabase = Supabase.instance.client;

  @override
  void initState() {
    super.initState();
    _loadSpecimenData();
  }

  void _loadSpecimenData() {
    _nameController.text = widget.specimen['name'] ?? '';
    _speciesController.text = widget.specimen['species'] ?? '';
    _isPublic = widget.specimen['is_public'] ?? false;
  }

  Future<void> _updateSpecimen() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    setState(() => _isLoading = true);

    try {
      // Update specimen in Supabase
      await _supabase
          .from('specimens')
          .update({
            'name': _nameController.text.trim(),
            'species': _speciesController.text.trim(),
            'is_public': _isPublic,
            'updated_at': DateTime.now().toIso8601String(),
          })
          .eq('id', widget.specimen['id']);

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Specimen updated successfully'),
            backgroundColor: Colors.green,
          ),
        );

        // Call the callback if provided
        if (widget.onSpecimenUpdated != null) {
          widget.onSpecimenUpdated!();
        }

        // Navigate back
        Navigator.of(context).pop();
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to update specimen: ${e.toString()}'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Edit Specimen')),
      body:
          _isLoading
              ? const Center(child: CircularProgressIndicator())
              : SingleChildScrollView(
                padding: const EdgeInsets.all(16),
                child: Form(
                  key: _formKey,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Image preview (read-only)
                      if (widget.specimen['image_url'] != null)
                        Center(
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(12),
                            child: Image.network(
                              widget.specimen['image_url'],
                              height: 200,
                              width: double.infinity,
                              fit: BoxFit.cover,
                              errorBuilder:
                                  (context, error, stackTrace) => Container(
                                    height: 200,
                                    color: Colors.grey[200],
                                    child: Center(
                                      child: Icon(
                                        Icons.image_not_supported,
                                        size: 64,
                                        color: Colors.grey[400],
                                      ),
                                    ),
                                  ),
                            ),
                          ),
                        ),

                      const SizedBox(height: 24),

                      // Name field
                      TextFormField(
                        controller: _nameController,
                        decoration: const InputDecoration(
                          labelText: 'Specimen Name',
                          border: OutlineInputBorder(),
                        ),
                        validator: (value) {
                          if (value == null || value.trim().isEmpty) {
                            return 'Please enter a name for your specimen';
                          }
                          return null;
                        },
                      ),

                      const SizedBox(height: 16),

                      // Species field
                      TextFormField(
                        controller: _speciesController,
                        decoration: const InputDecoration(
                          labelText: 'Species / Type',
                          border: OutlineInputBorder(),
                          helperText:
                              'E.g., Plant Cells, Animal Cells, Bacteria, etc.',
                        ),
                        validator: (value) {
                          if (value == null || value.trim().isEmpty) {
                            return 'Please enter the species or type';
                          }
                          return null;
                        },
                      ),

                      const SizedBox(height: 16),

                      // Public toggle
                      SwitchListTile(
                        title: const Text('Make Public'),
                        subtitle: const Text(
                          'Allow other users to see this specimen',
                        ),
                        value: _isPublic,
                        onChanged: (value) {
                          setState(() {
                            _isPublic = value;
                          });
                        },
                      ),

                      const SizedBox(height: 24),

                      // Submit button
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton(
                          onPressed: _updateSpecimen,
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 16),
                          ),
                          child: const Text('Update Specimen'),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
    );
  }

  @override
  void dispose() {
    _nameController.dispose();
    _speciesController.dispose();
    super.dispose();
  }
}
