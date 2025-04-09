// lib/screens/auth/policy_screen.dart
import 'package:flutter/material.dart';

class PolicyScreen extends StatelessWidget {
  const PolicyScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Privacy Policy & Terms'),
        backgroundColor: Colors.blue.shade900,
        foregroundColor: Colors.white,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'MicroVision Privacy Policy & Terms of Service',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: Color.fromARGB(255, 34, 46, 138),
                ),
              ),
              const Text(
                'Last Updated: March 31, 2025',
                style: TextStyle(fontSize: 14, color: Colors.grey),
              ),
              const SizedBox(height: 24),
              _buildSection(
                '1. Introduction',
                'Welcome to MicroVision, an AI-powered virtual microscopy platform for educational purposes. This document outlines how we handle your data and the terms governing your use of our services.',
              ),
              _buildSection('2. Information Collection and Use', '''
What We Collect:
• Account Information: Name, email, password, institution
• Content: Uploaded microscopy images, annotations, quiz responses
• Usage Data: Features used, learning progress, device information
• AI Processing Data: Images processed by our AI models

How We Use It:
• Provide virtual microscopy and AI annotation services
• Optimize for low-bandwidth environments
• Track learning progress and personalize experiences
• Improve our AI models and educational features
• Communicate necessary updates and respond to support requests
                '''),
              _buildSection('3. Data Protection', '''
• We encrypt data in transit and at rest
• We implement secure authentication through Supabase
• We optimize data storage for low-bandwidth conditions
• We provide offline functionality with secure local storage
• We retain your data only as long as necessary for educational purposes
                '''),
              _buildSection('4. Information Sharing', '''
We may share information:
• With educational institutions you're affiliated with (for progress tracking)
• With service providers helping us deliver our services
• When required by law or to protect rights and safety
• In anonymized form for educational research
                '''),
              _buildSection('5. Your Rights', '''
You can:
• Access, update, or delete your account information
• Control your communication preferences
• Request a copy of your data
• Use the service offline to manage bandwidth usage
                '''),
              _buildSection('6. Service Terms', '''
User Responsibilities:
• Provide accurate account information
• Maintain password confidentiality
• Use the service primarily for educational purposes
• Respect intellectual property rights

User Content:
• You retain ownership of content you upload
• You grant us license to use your content to provide the service
• You must have rights to content you share

Prohibited Activities:
• Violating laws or others' rights
• Attempting to interfere with the service
• Uploading malicious code
• Misusing the platform for non-educational purposes
                '''),
              _buildSection('7. AI-Generated Content', '''
• Some content is generated automatically by AI systems
• While we strive for accuracy, AI content should be verified for critical use
• AI features are designed to function in low-resource environments
                '''),
              _buildSection('8. Technical Requirements', '''
• Compatible device (mobile or computer)
• Internet connection (with offline capabilities for core features)
• We've optimized for low-bandwidth through data compression and offline functionality
                '''),
              _buildSection('9. Intellectual Property', '''
• The service and its original content are protected by intellectual property laws
• We utilize open-source components (licenses available upon request)
• Educational use of the platform can be referenced in academic work with attribution
                '''),
              _buildSection('10. Service Availability and Liability', '''
• We strive for reliability but cannot guarantee uninterrupted service
• The platform is provided for educational purposes and not a substitute for professional training
• Our liability is limited to the extent permitted by law
                '''),
              _buildSection('11. Updates to This Document', '''
We may update this document periodically. Continued use after changes constitutes acceptance of the updated terms.
                '''),
              _buildSection('12. Contact Us', '''
Email: m.moussa@alustudent.com
Mail: MicroVision Educational Technologies
Kigaali, Rwanda 
                '''),
              const SizedBox(height: 24),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () {
                    Navigator.pop(context);
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue.shade900,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                  ),
                  child: const Text('I Understand'),
                ),
              ),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSection(String title, String content) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: Color.fromARGB(255, 34, 46, 138),
            ),
          ),
          const SizedBox(height: 8),
          Text(content, style: const TextStyle(fontSize: 14, height: 1.5)),
        ],
      ),
    );
  }
}
