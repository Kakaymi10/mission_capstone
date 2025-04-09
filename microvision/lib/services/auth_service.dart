// lib/services/auth_service.dart

import 'package:supabase_flutter/supabase_flutter.dart';

class AuthService {
  final SupabaseClient _supabase;

  AuthService(this._supabase);

  Future<void> initialize() async {
    await Supabase.initialize(
      url: 'https://xmypecypllgrcgcehuli.supabase.co',
      anonKey:
          'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhteXBlY3lwbGxncmNnY2VodWxpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzk5NDIzNzEsImV4cCI6MjA1NTUxODM3MX0.WebWQtwO0uyTVJvEG9bK7Tax6d3UdNogqMmPjEuDXaU',
    );
  }

  Future<AuthResponse> signUp({
    required String email,
    required String password,
  }) async {
    return await _supabase.auth.signUp(email: email, password: password);
  }

  Future<AuthResponse> signIn({
    required String email,
    required String password,
  }) async {
    return await _supabase.auth.signInWithPassword(
      email: email,
      password: password,
    );
  }

  Future<void> signOut() async {
    await _supabase.auth.signOut();
  }

  bool get isAuthenticated => _supabase.auth.currentUser != null;

  User? get currentUser => _supabase.auth.currentUser;
}
