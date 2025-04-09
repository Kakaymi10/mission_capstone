import 'package:flutter/material.dart';
import 'package:microvision/screens/explore/canvas_screen.dart';
import 'package:microvision/screens/explore/explore_screen.dart';
import 'package:microvision/screens/profile/profile_screen.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'create_specimen_dialog.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0;
  final _supabase = Supabase.instance.client;
  List<Map<String, dynamic>> _specimens = [];
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _checkAuthAndLoadSpecimens();
  }

  Future<void> _loadSpecimens() async {
    setState(() => _isLoading = true);

    try {
      // Get current user's ID
      final userId = _supabase.auth.currentUser?.id;

      if (userId == null) {
        throw Exception('User not authenticated');
      }

      print("Current User ID: $userId"); // Debug print

      final response = await _supabase
          .from('specimens')
          .select()
          .eq('user_id', userId) // Add filter for current user
          .order('created_at', ascending: false)
          .limit(10);

      print("Supabase Response: $response"); // Debug print

      setState(() {
        _specimens = List<Map<String, dynamic>>.from(response);
      });
      print("SPECIMENS: $_specimens");
    } catch (e) {
      debugPrint('Error loading specimens: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to load specimens: ${e.toString()}'),
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

  Future<void> _checkAuthAndLoadSpecimens() async {
    final session = _supabase.auth.currentSession;

    if (session == null) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Please login to view specimens'),
            backgroundColor: Colors.orange,
          ),
        );
      }
      return;
    }

    await _loadSpecimens();
  }

  void _showCreateSpecimenDialog() {
    showDialog(
      context: context,
      builder:
          (context) => CreateSpecimenDialog(
            onSubmit: (name, species, isPublic, imageUrl) async {
              await _loadSpecimens(); // Reload specimens after creation
            },
          ),
    );
  }

  // Navigate to CanvasScreen with the selected specimen
  void _openSpecimenCanvas(Map<String, dynamic> specimen) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => ArcaneSpecimenViewer(magicalSpecimen: specimen),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'MicroVision',
          style: TextStyle(color: Colors.black87, fontWeight: FontWeight.bold),
        ),
        backgroundColor: Colors.white,
        elevation: 0,
        automaticallyImplyLeading: false, // Remove back button
        actions: [
          IconButton(
            icon: const Icon(
              Icons.notifications_outlined,
              color: Colors.black87,
            ),
            onPressed: () {
              // Handle notifications
              Navigator.pushNamed(context, '/canvas');
            },
          ),
          IconButton(
            icon: const Icon(Icons.person_outline, color: Colors.black87),
            onPressed: () async {},
          ),
        ],
      ),
      body: IndexedStack(
        index: _selectedIndex,
        children: [
          _buildMainContent(),
          const ExploreScreen(),
          const ProfileScreen(),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (index) {
          setState(() {
            _selectedIndex = index;
          });
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.home_outlined),
            selectedIcon: Icon(Icons.home),
            label: 'Home',
          ),
          NavigationDestination(
            icon: Icon(Icons.explore_outlined),
            selectedIcon: Icon(Icons.explore),
            label: 'Explore',
          ),
          NavigationDestination(
            icon: Icon(Icons.person_outline),
            selectedIcon: Icon(Icons.person),
            label: 'Profile',
          ),
        ],
      ),
    );
  }

  Widget _buildMainContent() {
    // Get screen width to adjust layouts
    final screenWidth = MediaQuery.of(context).size.width;

    // Determine if we're on a large screen
    final isLargeScreen = screenWidth > 600;

    // Adjust card and grid sizes based on screen width
    final specimenCardWidth = isLargeScreen ? 140.0 : 160.0;
    final gridCrossAxisCount = isLargeScreen ? 3 : 2;

    // Adjust content padding based on screen size
    final contentPadding =
        isLargeScreen
            ? const EdgeInsets.symmetric(horizontal: 24, vertical: 16)
            : const EdgeInsets.all(16);

    return RefreshIndicator(
      onRefresh: _loadSpecimens,
      child: SingleChildScrollView(
        padding: contentPadding,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Welcome Section - Responsive with max width
            // Welcome Section - Centered for all screen sizes
            Center(
              child: Container(
                padding: const EdgeInsets.all(20),
                constraints: BoxConstraints(
                  maxWidth: isLargeScreen ? 800 : double.infinity,
                ),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Colors.blue.shade800, Colors.blue.shade600],
                  ),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Welcome back, ${_supabase.auth.currentUser?.email?.split('@')[0] ?? 'User'}!',
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 8),
                          const Text(
                            'Continue your microscopic journey',
                            style: TextStyle(
                              color: Colors.white70,
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const Icon(Icons.science, color: Colors.white, size: 48),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Recent Specimens Section - with wrapper for alignment
            Center(
              child: ConstrainedBox(
                constraints: BoxConstraints(
                  maxWidth: isLargeScreen ? 800 : double.infinity,
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text(
                          'Recent Specimens',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        TextButton(
                          onPressed: () {
                            setState(
                              () => _selectedIndex = 1,
                            ); // Switch to Explore tab
                          },
                          child: const Text('See All'),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    _buildRecentSpecimens(specimenCardWidth),
                    const SizedBox(height: 24),

                    // Quick Actions - Responsive grid
                    const Text(
                      'Quick Actions',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16),

                    // Replace the GridView.count in the _buildMainContent method with this code
                    GridView.count(
                      shrinkWrap: true,
                      physics: const NeverScrollableScrollPhysics(),
                      crossAxisCount: gridCrossAxisCount,
                      mainAxisSpacing: 16,
                      crossAxisSpacing: 16,
                      childAspectRatio:
                          isLargeScreen
                              ? 1.3
                              : 1.0, // Make cards less tall on larger screens
                      children: [
                        _buildQuickActionCard(
                          'New Specimen',
                          Icons.add_circle_outline,
                          Colors.green,
                          _showCreateSpecimenDialog,
                        ),
                        _buildQuickActionCard(
                          'Explore',
                          Icons.explore_outlined,
                          Colors.orange,
                          () {
                            setState(
                              () => _selectedIndex = 1,
                            ); // Switch to Explore tab
                          },
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRecentSpecimens(double cardWidth) {
    if (_isLoading) {
      return const SizedBox(
        height: 180,
        child: Center(child: CircularProgressIndicator()),
      );
    }

    if (_specimens.isEmpty) {
      return Card(
        child: Container(
          height: 180,
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.science_outlined, size: 48, color: Colors.grey),
              const SizedBox(height: 16),
              const Text(
                'No specimens yet',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              ElevatedButton(
                onPressed: _showCreateSpecimenDialog,
                child: const Text('Create Your First Specimen'),
              ),
            ],
          ),
        ),
      );
    }

    return SizedBox(
      height: 180,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: _specimens.length,
        itemBuilder: (context, index) {
          final specimen = _specimens[index];
          return GestureDetector(
            onTap: () => _openSpecimenCanvas(specimen),
            child: Card(
              margin: const EdgeInsets.only(right: 16),
              child: Container(
                width: cardWidth, // Use responsive width
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    specimen['image_url'] != null
                        ? ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: Image.network(
                            specimen['image_url'],
                            height: 100,
                            width: double.infinity,
                            fit: BoxFit.cover,
                            errorBuilder:
                                (context, error, stackTrace) =>
                                    _buildPlaceholderImage(),
                          ),
                        )
                        : _buildPlaceholderImage(),
                    const SizedBox(height: 8),
                    Text(
                      specimen['name'] ?? 'Unnamed Specimen',
                      style: const TextStyle(fontWeight: FontWeight.bold),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    Text(
                      specimen['species'] ?? 'Unknown Species',
                      style: TextStyle(
                        color: Colors.grey.shade600,
                        fontSize: 12,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildPlaceholderImage() {
    return Container(
      height: 100,
      decoration: BoxDecoration(
        color: Colors.grey.shade200,
        borderRadius: BorderRadius.circular(8),
      ),
      child: const Center(child: Icon(Icons.image, size: 32)),
    );
  }

  Widget _buildQuickActionCard(
    String title,
    IconData icon,
    Color color,
    VoidCallback onTap,
  ) {
    return InkWell(
      onTap: onTap,
      child: Card(
        child: Container(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, size: 32, color: color),
              const SizedBox(height: 8),
              Text(
                title,
                style: const TextStyle(fontWeight: FontWeight.bold),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
