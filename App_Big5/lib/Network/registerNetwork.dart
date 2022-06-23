import 'dart:convert';
import '../utill/Verification.dart';
import 'package:http/http.dart' as http;

class registerNetwork {
  Future postRegister(String phoneNumber, String fullName,String displayName,
      String password) async {
    var Key = Validation.Key;
    var url = Uri.parse("https://192.168.1.9:5000/users/register");
    var response = await http.post(url, body: {
      "phoneNumber": phoneNumber,
      "fullName": fullName,
      "displayName":displayName,
      "key": Key,
      "password": password
    });
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    }
    else {
      throw Exception("Error");
    }
  }
}