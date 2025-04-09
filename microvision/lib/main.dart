// lib/main.dart

import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:microvision/config/routes.dart';
import 'package:microvision/screens/landing/landing_screen.dart';

class MicroVisionApp extends StatelessWidget {
  const MicroVisionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'MicroVision',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        // Add more theme configurations here
        scaffoldBackgroundColor: Colors.white,
        appBarTheme: const AppBarTheme(
          elevation: 0,
          backgroundColor: Colors.white,
          iconTheme: IconThemeData(color: Colors.black),
        ),
      ),
      // Set initial route
      initialRoute: '/',
      // Use the routes from routes.dart
      routes: routes,
      // You can add onGenerateRoute for dynamic routing if needed
      onGenerateRoute: (settings) {
        // Handle unknown routes
        return MaterialPageRoute(builder: (context) => const LandingPage());
      },
    );
  }
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  try {
    await Supabase.initialize(
      url: 'https://xmypecypllgrcgcehuli.supabase.co',
      anonKey:
          'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhteXBlY3lwbGxncmNnY2VodWxpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzk5NDIzNzEsImV4cCI6MjA1NTUxODM3MX0.WebWQtwO0uyTVJvEG9bK7Tax6d3UdNogqMmPjEuDXaU',
      debug: true, // Enable debug mode for more detailed logs
    );
    print('Supabase initialized successfully');
  } catch (e) {
    print('Error initializing Supabase: $e');
    // You might want to show a user-friendly error message here
  }

  runApp(const MicroVisionApp());
}
