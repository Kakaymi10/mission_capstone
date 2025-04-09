// Import the CreateSpecimenDialog
import 'package:flutter/material.dart';
import 'package:microvision/screens/explore/canvas_screen.dart';
import 'package:microvision/screens/explore/edit_specimen_screen.dart';
import 'package:microvision/screens/home/create_specimen_dialog.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class ExploreScreen extends StatefulWidget {
  const ExploreScreen({super.key});

  @override
  _ExploreScreenState createState() => _ExploreScreenState();
}

class _ExploreScreenState extends State<ExploreScreen> {
  final _searchController = TextEditingController();
  String _selectedCategory = 'All';
  String _selectedFilter = 'All';
  bool _isLoading = false;
  final bool _isSearching = false;
  List<Map<String, dynamic>> _specimens = [];
  final _supabase = Supabase.instance.client;

  final List<String> _categories = [
    'All',
    'plant cells',
    'animal cells',
    'bacteria',
    'fungi',
    'tissue samples',
  ];

  final List<String> _filters = ['All', 'Mine'];

  @override
  void initState() {
    super.initState();
    _loadSpecimens();
  }

  Future<void> _loadSpecimens() async {
    setState(() => _isLoading = true);

    try {
      // Start building our query
      var supabaseQuery = _supabase.from('specimens').select();

      // Apply Mine/All filter
      if (_selectedFilter == 'Mine') {
        final userId = _supabase.auth.currentUser?.id;
        if (userId != null) {
          supabaseQuery = supabaseQuery.eq('user_id', userId);
        } else {
          // If no user is logged in but "Mine" is selected, return empty list
          setState(() {
            _specimens = [];
            _isLoading = false;
          });
          _showError('Please log in to view your specimens');
          return;
        }
      } else {
        // Only show public specimens when "All" is selected
        supabaseQuery = supabaseQuery.eq('is_public', true);
      }

      // Apply category filter if a specific category is selected
      if (_selectedCategory != 'All') {
        supabaseQuery = supabaseQuery.eq('species', _selectedCategory);
      }

      // Apply search filter if there's text in the search box
      if (_searchController.text.isNotEmpty) {
        supabaseQuery = supabaseQuery.ilike(
          'name',
          '%${_searchController.text}%',
        );
      }

      // Order by newest first
      final response = await supabaseQuery.order(
        'created_at',
        ascending: false,
      );

      setState(() {
        _specimens = List<Map<String, dynamic>>.from(response);
      });
    } catch (e) {
      debugPrint('Error loading specimens: $e');
      _showError('Failed to load specimens');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // Show the create specimen dialog
  void _showCreateSpecimenDialog() {
    showDialog(
      context: context,
      builder:
          (context) => CreateSpecimenDialog(
            onSubmit: (name, species, isPublic, imageUrl) async {
              // Reload specimens after creation
              await _loadSpecimens();
            },
          ),
    );
  }

  // Navigate to the edit screen when edit is pressed
  void _navigateToEditScreen(Map<String, dynamic> specimen) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder:
            (context) => EditSpecimenScreen(
              specimen: specimen,
              onSpecimenUpdated: () {
                // Reload specimens when returning from edit screen
                _loadSpecimens();
              },
            ),
      ),
    );
  }

  // Handle specimen deletion
  Future<void> _deleteSpecimen(Map<String, dynamic> specimen) async {
    setState(() => _isLoading = true);

    try {
      // Delete from Supabase
      await _supabase.from('specimens').delete().eq('id', specimen['id']);

      // Refresh the list
      await _loadSpecimens();

      // Show success message
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Specimen deleted successfully'),
            backgroundColor: Colors.green,
          ),
        );
      }
    } catch (e) {
      debugPrint('Error deleting specimen: $e');
      _showError('Failed to delete specimen');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  void _showError(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red),
    );
  }

  @override
  Widget build(BuildContext context) {
    // Get screen width to determine layout
    final screenWidth = MediaQuery.of(context).size.width;

    // Determine responsive values based on screen size
    final bool isLargeScreen = screenWidth > 600;
    final int gridCrossAxisCount =
        screenWidth > 900
            ? 4
            : isLargeScreen
            ? 3
            : 2;
    final double childAspectRatio = isLargeScreen ? 0.85 : 0.75;

    // Content max width for large screens
    final double contentMaxWidth = isLargeScreen ? 1200 : double.infinity;
    final EdgeInsetsGeometry contentPadding =
        isLargeScreen
            ? const EdgeInsets.symmetric(horizontal: 24, vertical: 16)
            : const EdgeInsets.all(16);

    return Column(
      children: [
        // Search and filter section with responsive width constraints
        Center(
          child: ConstrainedBox(
            constraints: BoxConstraints(maxWidth: contentMaxWidth),
            child: Padding(
              padding: contentPadding,
              child: Column(
                children: [
                  // Search Bar
                  TextField(
                    controller: _searchController,
                    decoration: InputDecoration(
                      hintText: 'Search specimens...',
                      prefixIcon: const Icon(Icons.search),
                      suffixIcon:
                          _searchController.text.isNotEmpty
                              ? IconButton(
                                icon: const Icon(Icons.clear),
                                onPressed: () {
                                  _searchController.clear();
                                  _loadSpecimens();
                                },
                              )
                              : null,
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(30),
                      ),
                      filled: true,
                      fillColor: Colors.grey[100],
                    ),
                    onChanged: (value) {
                      if (value.length >= 2 || value.isEmpty) {
                        _loadSpecimens();
                      }
                    },
                  ),
                  const SizedBox(height: 16),
                  // Filters
                  Row(
                    children: [
                      Expanded(
                        child: SizedBox(
                          height: 40,
                          child: ListView.builder(
                            scrollDirection: Axis.horizontal,
                            itemCount: _filters.length,
                            itemBuilder: (context, index) {
                              return Padding(
                                padding: const EdgeInsets.only(right: 8),
                                child: FilterChip(
                                  label: Text(_filters[index]),
                                  selected: _selectedFilter == _filters[index],
                                  onSelected: (selected) {
                                    setState(() {
                                      _selectedFilter = _filters[index];
                                    });
                                    _loadSpecimens();
                                  },
                                ),
                              );
                            },
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  // Categories
                  SizedBox(
                    height: 40,
                    child: ListView.builder(
                      scrollDirection: Axis.horizontal,
                      itemCount: _categories.length,
                      itemBuilder: (context, index) {
                        return Padding(
                          padding: const EdgeInsets.only(right: 8),
                          child: ChoiceChip(
                            label: Text(_categories[index]),
                            selected: _selectedCategory == _categories[index],
                            onSelected: (selected) {
                              setState(() {
                                _selectedCategory = _categories[index];
                              });
                              _loadSpecimens();
                            },
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

        // Main content section
        if (_isLoading || _isSearching)
          const Expanded(child: Center(child: CircularProgressIndicator()))
        else if (_specimens.isEmpty)
          Expanded(
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.search_off, size: 64, color: Colors.grey[400]),
                  const SizedBox(height: 16),
                  Text(
                    _searchController.text.isNotEmpty
                        ? 'No specimens found for "${_searchController.text}"'
                        : _selectedFilter == 'Mine'
                        ? 'You have no specimens yet'
                        : 'No specimens available',
                    style: TextStyle(fontSize: 16, color: Colors.grey[600]),
                  ),
                  if (_selectedFilter == 'Mine')
                    Padding(
                      padding: const EdgeInsets.only(top: 16),
                      child: ElevatedButton(
                        onPressed: _showCreateSpecimenDialog,
                        child: const Text('Create New Specimen'),
                      ),
                    ),
                ],
              ),
            ),
          )
        else
          Expanded(
            child: Center(
              child: ConstrainedBox(
                constraints: BoxConstraints(maxWidth: contentMaxWidth),
                child: GridView.builder(
                  padding: contentPadding,
                  gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: gridCrossAxisCount,
                    crossAxisSpacing: 16,
                    mainAxisSpacing: 16,
                    childAspectRatio: childAspectRatio,
                  ),
                  itemCount: _specimens.length,
                  itemBuilder: (context, index) {
                    final specimen = _specimens[index];
                    final userId = _supabase.auth.currentUser?.id;
                    final isMine = specimen['user_id'] == userId;

                    return SpecimenCard(
                      title: specimen['name'] ?? 'Unnamed Specimen',
                      description: specimen['species'] ?? 'No description',
                      imageUrl: specimen['image_url'],
                      isMine: isMine,
                      specimen: specimen,
                      onTap: () {
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder:
                                (context) => ArcaneSpecimenViewer(
                                  magicalSpecimen: specimen,
                                ),
                          ),
                        );
                      },
                      onEdit: isMine ? _navigateToEditScreen : null,
                      onDelete: isMine ? _deleteSpecimen : null,
                      isCompact: isLargeScreen && gridCrossAxisCount > 2,
                    );
                  },
                ),
              ),
            ),
          ),
      ],
    );
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }
}

class SpecimenCard extends StatelessWidget {
  final String title;
  final String description;
  final String? imageUrl;
  final VoidCallback onTap;
  final bool isMine;
  final Map<String, dynamic> specimen;
  final Function(Map<String, dynamic>)? onEdit;
  final Function(Map<String, dynamic>)? onDelete;
  final bool isCompact;

  const SpecimenCard({
    super.key,
    required this.title,
    required this.description,
    this.imageUrl,
    required this.onTap,
    this.isMine = false,
    required this.specimen,
    this.onEdit,
    this.onDelete,
    this.isCompact = false,
  });

  @override
  Widget build(BuildContext context) {
    // Adjust image height based on card size
    final double imageHeight = isCompact ? 120.0 : 150.0;

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Stack(
              children: [
                ClipRRect(
                  borderRadius: const BorderRadius.vertical(
                    top: Radius.circular(12),
                  ),
                  child: _buildImage(imageHeight),
                ),
                if (isMine)
                  Positioned(
                    top: 8,
                    right: 8,
                    child: Row(
                      children: [
                        // Edit button
                        Container(
                          margin: const EdgeInsets.only(right: 4),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.9),
                            shape: BoxShape.circle,
                          ),
                          child: IconButton(
                            icon: const Icon(Icons.edit, size: 16),
                            color: Colors.blue,
                            constraints: const BoxConstraints(
                              minWidth: 32,
                              minHeight: 32,
                            ),
                            padding: EdgeInsets.zero,
                            onPressed: () {
                              if (onEdit != null) onEdit!(specimen);
                            },
                          ),
                        ),
                        // Delete button
                        Container(
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.9),
                            shape: BoxShape.circle,
                          ),
                          child: IconButton(
                            icon: const Icon(Icons.delete, size: 16),
                            color: Colors.red,
                            constraints: const BoxConstraints(
                              minWidth: 32,
                              minHeight: 32,
                            ),
                            padding: EdgeInsets.zero,
                            onPressed: () {
                              if (onDelete != null) {
                                _showDeleteConfirmation(context);
                              }
                            },
                          ),
                        ),
                      ],
                    ),
                  ),
                if (isMine)
                  Positioned(
                    top: 8,
                    left: 8,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.blue.withOpacity(0.9),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Text(
                        'Mine',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ),
              ],
            ),
            Padding(
              padding:
                  isCompact
                      ? const EdgeInsets.all(8)
                      : const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      fontSize: isCompact ? 14 : 16,
                      fontWeight: FontWeight.bold,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  SizedBox(height: isCompact ? 2 : 4),
                  Text(
                    description,
                    style: TextStyle(
                      color: Colors.grey[600],
                      fontSize: isCompact ? 12 : 14,
                    ),
                    maxLines: isCompact ? 1 : 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // Show delete confirmation dialog
  void _showDeleteConfirmation(BuildContext context) {
    showDialog(
      context: context,
      builder:
          (context) => AlertDialog(
            title: const Text('Delete Specimen'),
            content: const Text(
              'Are you sure you want to delete this specimen? This action cannot be undone.',
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text('Cancel'),
              ),
              TextButton(
                onPressed: () {
                  Navigator.of(context).pop();
                  if (onDelete != null) onDelete!(specimen);
                },
                child: const Text(
                  'Delete',
                  style: TextStyle(color: Colors.red),
                ),
              ),
            ],
          ),
    );
  }

  Widget _buildImage(double height) {
    if (imageUrl == null || imageUrl!.isEmpty) {
      return _buildPlaceholder(height);
    }

    return Image.network(
      imageUrl!,
      height: height,
      width: double.infinity,
      fit: BoxFit.cover,
      loadingBuilder: (context, child, loadingProgress) {
        if (loadingProgress == null) return child;
        return Container(
          height: height,
          color: Colors.grey[200],
          child: Center(
            child: CircularProgressIndicator(
              value:
                  loadingProgress.expectedTotalBytes != null
                      ? loadingProgress.cumulativeBytesLoaded /
                          loadingProgress.expectedTotalBytes!
                      : null,
            ),
          ),
        );
      },
      errorBuilder: (context, error, stackTrace) => _buildPlaceholder(height),
    );
  }

  Widget _buildPlaceholder(double height) {
    return Container(
      height: height,
      color: Colors.grey[200],
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.image_outlined,
              size: isCompact ? 36 : 48,
              color: Colors.grey[400],
            ),
            if (!isCompact) const SizedBox(height: 8),
            Text(
              'No Image',
              style: TextStyle(
                color: Colors.grey[600],
                fontSize: isCompact ? 10 : 12,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
