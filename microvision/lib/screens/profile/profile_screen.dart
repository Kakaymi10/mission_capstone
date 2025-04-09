// lib/screens/profile/profile_screen.dart

import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final _supabase = Supabase.instance.client;
  int _specimenCount = 0;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _loadSpecimenCount();
  }

  Future<void> _loadSpecimenCount() async {
    setState(() => _isLoading = true);
    try {
      final userId = _supabase.auth.currentUser?.id;

      if (userId != null) {
        final response = await _supabase
            .from('specimens')
            .select('id')
            .eq('user_id', userId);

        setState(() {
          _specimenCount = response.length;
        });
      }
    } catch (e) {
      debugPrint('Error loading specimen count: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    // Get screen width to determine layout
    final screenWidth = MediaQuery.of(context).size.width;
    final isLargeScreen = screenWidth > 600;

    return Center(
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxWidth: isLargeScreen ? 600 : double.infinity,
        ),
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // Profile Header
            Column(
              children: [
                CircleAvatar(
                  radius: 50,
                  backgroundColor: Colors.blue.shade100,
                  child: Text(
                    _getInitials(),
                    style: const TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                      color: Colors.blue,
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                Text(
                  _supabase.auth.currentUser?.email ?? 'User',
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 32),

            // Stats Section - Only Specimens Count
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.shade200,
                    blurRadius: 10,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child:
                  _isLoading
                      ? const Center(child: CircularProgressIndicator())
                      : Column(
                        children: [
                          Text(
                            _specimenCount.toString(),
                            style: const TextStyle(
                              fontSize: 36,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          Text(
                            'Specimens',
                            style: TextStyle(
                              color: Colors.grey.shade600,
                              fontSize: 16,
                            ),
                          ),
                        ],
                      ),
            ),
            const SizedBox(height: 24),

            // Only Logout Menu Item
            _buildMenuItem(
              icon: Icons.logout,
              title: 'Logout',
              onTap: () async {
                await _supabase.auth.signOut();
                if (mounted) {
                  Navigator.of(context).pushReplacementNamed('/login');
                }
              },
              textColor: Colors.red,
            ),
          ],
        ),
      ),
    );
  }

  String _getInitials() {
    final email = _supabase.auth.currentUser?.email ?? '';
    if (email.isEmpty) return 'U';
    return email[0].toUpperCase();
  }

  Widget _buildMenuItem({
    required IconData icon,
    required String title,
    VoidCallback? onTap,
    Color? textColor,
  }) {
    return ListTile(
      leading: Icon(icon, color: textColor),
      title: Text(
        title,
        style: TextStyle(color: textColor, fontWeight: FontWeight.w500),
      ),
      trailing: const Icon(Icons.chevron_right),
      onTap: onTap,
    );
  }
}
