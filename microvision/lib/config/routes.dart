// lib/config/routes.dart

import 'package:flutter/material.dart';
import 'package:microvision/screens/auth/policy_screen.dart';
import 'package:microvision/screens/home/home_screen.dart';
import 'package:microvision/screens/landing/landing_screen.dart';
import 'package:microvision/screens/auth/login_screen.dart';
import 'package:microvision/screens/auth/signup_screen.dart';

final Map<String, WidgetBuilder> routes = {
  '/': (context) => const LandingPage(),
  '/login': (context) => const LoginScreen(),
  '/signup': (context) => const SignupScreen(),
  '/home': (context) => const HomeScreen(),
  '/policy': (context) => const PolicyScreen(),
};
