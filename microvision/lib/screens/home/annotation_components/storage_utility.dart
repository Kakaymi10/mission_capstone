import 'dart:convert';
import 'dart:ui';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Represents a segmentation mask with its properties
class SegmentationMask {
  final String id;
  final Offset? clickPoint;
  final bool isForeground;
  final String note;
  final List<int> maskImage;

  SegmentationMask({
    required this.id,
    this.clickPoint,
    required this.isForeground,
    this.note = '',
    required this.maskImage,
  });
}

/// A utility class to save and retrieve segmentation data in local storage
class SegmentationStorageUtil {
  static const String _storageKeyPrefix = 'segmentation_data_';

  /// Saves segmentation data for an image
  static Future<bool> saveSegmentationData({
    required String imageUrl,
    required List<Map<String, dynamic>> points,
  }) async {
    try {
      final prefs = await SharedPreferences.getInstance();

      // Create a storage key from the image URL
      final storageKey = _generateStorageKey(imageUrl);

      // Convert points to JSON string
      final jsonData = jsonEncode({
        'imageUrl': imageUrl,
        'points': points,
        'timestamp': DateTime.now().millisecondsSinceEpoch,
      });

      // Save to shared preferences
      final result = await prefs.setString(storageKey, jsonData);

      if (kDebugMode) {
        print('Saved segmentation data for: $imageUrl');
        print('Storage key: $storageKey');
      }

      return result;
    } catch (e) {
      if (kDebugMode) {
        print('Error saving segmentation data: $e');
      }
      return false;
    }
  }

  /// Retrieves segmentation data for an image
  static Future<Map<String, dynamic>?> getSegmentationData(
    String imageUrl,
  ) async {
    try {
      final prefs = await SharedPreferences.getInstance();

      // Create a storage key from the image URL
      final storageKey = _generateStorageKey(imageUrl);

      // Get data from shared preferences
      final jsonData = prefs.getString(storageKey);

      if (jsonData == null) {
        return null;
      }

      // Parse the JSON data
      return jsonDecode(jsonData);
    } catch (e) {
      if (kDebugMode) {
        print('Error retrieving segmentation data: $e');
      }
      return null;
    }
  }

  /// Deletes segmentation data for an image
  static Future<bool> deleteSegmentationData(String imageUrl) async {
    try {
      final prefs = await SharedPreferences.getInstance();

      // Create a storage key from the image URL
      final storageKey = _generateStorageKey(imageUrl);

      // Remove data from shared preferences
      final result = await prefs.remove(storageKey);

      return result;
    } catch (e) {
      if (kDebugMode) {
        print('Error deleting segmentation data: $e');
      }
      return false;
    }
  }

  /// Lists all saved segmentation data
  static Future<List<String>> getAllImageUrlsWithSegmentationData() async {
    try {
      final prefs = await SharedPreferences.getInstance();

      // Get all keys from shared preferences
      final allKeys = prefs.getKeys();

      // Filter keys to only include segmentation data keys
      final segmentationKeys =
          allKeys.where((key) => key.startsWith(_storageKeyPrefix)).toList();

      // Extract image URLs from the keys
      final imageUrls =
          segmentationKeys.map((key) => _extractImageUrlFromKey(key)).toList();

      return imageUrls;
    } catch (e) {
      if (kDebugMode) {
        print('Error listing segmentation data: $e');
      }
      return [];
    }
  }

  /// Converts the image URL to a storage key
  static String _generateStorageKey(String imageUrl) {
    // Create a hash of the URL to use as the key
    final urlHash = imageUrl.hashCode.toString();
    return '$_storageKeyPrefix$urlHash';
  }

  /// Extracts the original image URL from a storage key
  /// This is a placeholder since we can't reverse the hash
  static String _extractImageUrlFromKey(String storageKey) {
    // In a real implementation, you would need to store a mapping
    // between keys and URLs to reverse this lookup
    return storageKey.replaceFirst(_storageKeyPrefix, '');
  }
}

/// Extension method to convert a list of SegmentationMask objects to a
/// format that can be saved in local storage
extension SegmentationMaskListExtension on List<SegmentationMask> {
  List<Map<String, dynamic>> toStorageFormat() {
    return map(
      (mask) => {
        'id': mask.id,
        'clickX': mask?.clickPoint?.dx,
        'clickY': mask?.clickPoint?.dy,
        'isForeground': mask.isForeground,
        'note': mask.note,
        // We don't save the full mask image to local storage as it would be too large
        // Just store a reference to it if needed
        'hasMaskImage': mask.maskImage.isNotEmpty,
      },
    ).toList();
  }
}
