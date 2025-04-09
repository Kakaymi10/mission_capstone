import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:supabase_flutter/supabase_flutter.dart';

class AnnotationService {
  final SupabaseClient _supabase = Supabase.instance.client;

  // Get the appropriate API URL based on platform
  String get _apiBaseUrl {
    if (kIsWeb) {
      return 'http://127.0.0.1:8000'; // Web URL
    } else {
      return 'http://10.0.2.2:8000'; // Mobile URL (Android emulator)
    }
  }

  // Fetch annotations for a specific specimen by image URL
  Future<Map<String, dynamic>> fetchAnnotations(String imageUrl) async {
    try {
      print('Fetching annotations for image URL: $imageUrl');

      // Modified query to avoid the .single() operator which was causing errors
      // Use .select() and handle possible multiple results
      final List<dynamic> response = await _supabase
          .from('image_annotations')
          .select()
          .eq('image_url', imageUrl);

      print('Supabase response: $response');

      if (response.isNotEmpty) {
        // Take the first annotation if multiple exist
        final annotation = response[0];

        // Get the raw_annotations field which contains the bounding box coordinates
        final rawAnnotations = annotation['raw_annotations'];
        print('Raw annotations from DB: $rawAnnotations');

        if (rawAnnotations != null) {
          // The raw_annotations field is a JSON string in your database
          // We need to parse it to get the list of annotations
          if (rawAnnotations is String) {
            try {
              // Parse the JSON string to a List - ROBUST FIX for double-escaped JSON
              // The string is likely double-encoded in the database
              String processedString = rawAnnotations;

              // 1. Remove the outer quotes if they exist
              if (processedString.startsWith('"') &&
                  processedString.endsWith('"')) {
                processedString = processedString.substring(
                  1,
                  processedString.length - 1,
                );
              }

              // 2. Replace all escaped quotes with regular quotes
              processedString = processedString.replaceAll('\\"', '"');

              // 3. Now parse the cleaned string
              final List<dynamic> parsedAnnotations = jsonDecode(
                processedString,
              );
              print(
                'Successfully parsed annotations: ${parsedAnnotations.length} items',
              );

              // Transform the annotations to the required format (x1,y1,x2,y2)
              final transformedAnnotations =
                  parsedAnnotations.map((annotation) {
                    // Extract coordinates and dimensions
                    final double x =
                        (annotation['x'] is num)
                            ? annotation['x'].toDouble()
                            : double.parse(annotation['x'].toString());
                    final double y =
                        (annotation['y'] is num)
                            ? annotation['y'].toDouble()
                            : double.parse(annotation['y'].toString());
                    final double width =
                        (annotation['width'] is num)
                            ? annotation['width'].toDouble()
                            : double.parse(annotation['width'].toString());
                    final double height =
                        (annotation['height'] is num)
                            ? annotation['height'].toDouble()
                            : double.parse(annotation['height'].toString());

                    // Calculate the bottom-right coordinates (x2, y2)
                    final double x2 = x + width;
                    final double y2 = y + height;

                    // Return the transformed annotation with the required format
                    return {
                      'id': annotation['id'],
                      'x1': x,
                      'y1': y,
                      'x2': x2,
                      'y2': y2,
                      'note': annotation['note'],
                    };
                  }).toList();

              print('Transformed annotations: $transformedAnnotations');
              return {'success': true, 'annotations': transformedAnnotations};
            } catch (e) {
              print('Error parsing raw_annotations JSON: $e');
              return {
                'success': false,
                'error': 'Invalid annotation format: $e',
              };
            }
          } else if (rawAnnotations is List) {
            // If it's already a List, transform it to the required format
            final transformedAnnotations =
                rawAnnotations.map((annotation) {
                  final double x =
                      (annotation['x'] is num)
                          ? annotation['x'].toDouble()
                          : double.parse(annotation['x'].toString());
                  final double y =
                      (annotation['y'] is num)
                          ? annotation['y'].toDouble()
                          : double.parse(annotation['y'].toString());
                  final double width =
                      (annotation['width'] is num)
                          ? annotation['width'].toDouble()
                          : double.parse(annotation['width'].toString());
                  final double height =
                      (annotation['height'] is num)
                          ? annotation['height'].toDouble()
                          : double.parse(annotation['height'].toString());

                  final double x2 = x + width;
                  final double y2 = y + height;

                  return {
                    'id': annotation['id'],
                    'x1': x,
                    'y1': y,
                    'x2': x2,
                    'y2': y2,
                    'note': annotation['note'],
                  };
                }).toList();

            return {'success': true, 'annotations': transformedAnnotations};
          } else {
            print(
              'Unexpected raw_annotations type: ${rawAnnotations.runtimeType}',
            );
            return {'success': false, 'error': 'Unexpected annotation format'};
          }
        }
      }

      // If no annotations found in database, try the API endpoint
      // Modified to use consistent Accept header to avoid 406 errors
      final apiResponse = await http.get(
        Uri.parse('$_apiBaseUrl/annotations'),
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'image_url': imageUrl, // Pass image URL as header
        },
      );

      print('API response status: ${apiResponse.statusCode}');
      print('API response body: ${apiResponse.body}');

      if (apiResponse.statusCode == 200) {
        return {
          'success': true,
          'annotations': json.decode(apiResponse.body)['annotations'],
        };
      } else {
        throw Exception(
          'Failed to load annotations: ${apiResponse.statusCode}',
        );
      }
    } catch (e) {
      print('Error fetching annotations: $e');
      return {'success': false, 'error': e.toString()};
    }
  }

  // Segment image with annotations
  Future<Uint8List?> segmentImage(
    String imageUrl,
    List<Map<String, dynamic>> annotations,
  ) async {
    try {
      // Format bounding boxes for API call: "x1,y1,x2,y2;x1,y1,x2,y2"
      final String bboxesString = annotations
          .map((box) => "${box['x1']},${box['y1']},${box['x2']},${box['y2']}")
          .join(';');

      print('Bounding boxes: $bboxesString');

      // Download the image from the URL
      final imageResponse = await http.get(Uri.parse(imageUrl));
      if (imageResponse.statusCode != 200) {
        throw Exception('Failed to fetch image: ${imageResponse.statusCode}');
      }

      // Create multipart request to segment endpoint with proper headers
      final segmentRequest = http.MultipartRequest(
        'POST',
        Uri.parse('$_apiBaseUrl/segment/'),
      );

      segmentRequest.headers.addAll({
        'Accept': 'image/*', // Explicitly accept image response
      });

      // Add the image file
      segmentRequest.files.add(
        http.MultipartFile.fromBytes(
          'image',
          imageResponse.bodyBytes,
          filename: 'specimen_image.jpg',
        ),
      );

      // Add the bounding boxes parameter
      segmentRequest.fields['bboxes'] = bboxesString;

      print('Sending segmentation request...');

      // Send the segmentation request
      final streamedResponse = await segmentRequest.send();
      final segmentResponse = await http.Response.fromStream(streamedResponse);

      print('Segment response status: ${segmentResponse.statusCode}');

      if (segmentResponse.statusCode != 200) {
        throw Exception(
          'Failed to segment image: ${segmentResponse.statusCode}',
        );
      }

      return segmentResponse.bodyBytes;
    } catch (e) {
      print('Error during segmentation: $e');
      return null;
    }
  }

  Future<Map<String, dynamic>> segmentImageFromClick(
    String imageUrl,
    double pointX,
    double pointY,
    int pointType, {
    double conf = 0.4,
    double iou = 0.9, required int clickRadius, required int radius,
  }) async {
    try {
      // Download the image to get bytes
      final response = await http.get(Uri.parse(imageUrl));

      if (response.statusCode != 200) {
        return {
          'success': false,
          'error': 'Failed to fetch image: ${response.statusCode}',
        };
      }

      // Create multipart request
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('http://127.0.0.1:8000/segment_click/'),
      );

      // Add the image file
      request.files.add(
        http.MultipartFile.fromBytes(
          'image',
          response.bodyBytes,
          filename: 'image.jpg',
        ),
      );

      // Add click parameters
      request.fields['point_x'] = pointX.toString();
      request.fields['point_y'] = pointY.toString();
      request.fields['point_type'] = pointType.toString(); // 1 for foreground
      request.fields['conf'] = conf.toString();
      request.fields['iou'] = iou.toString();

      // Send the request
      final streamedResponse = await request.send();
      final segmentResponse = await http.Response.fromStream(streamedResponse);

      if (segmentResponse.statusCode != 200) {
        return {
          'success': false,
          'error': 'Segmentation API error: ${segmentResponse.statusCode}',
          'message': segmentResponse.body,
        };
      }

      // Return the segmented image response
      return {'success': true, 'data': segmentResponse.bodyBytes};
    } catch (e) {
      return {'success': false, 'error': 'Error during segmentation: $e'};
    }
  }
}
