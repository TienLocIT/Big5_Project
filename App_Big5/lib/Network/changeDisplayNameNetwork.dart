import 'dart:convert';

import '../utill/Verification.dart';
import "package:http/http.dart" as http;
class changeDisplayNameNetwork{
  Future changeDisplayNameMethod(String displayName,String phoneNumber) async {
    var url="https://192.168.1.9:5000/users/changeDisplayName";
    var key=Validation.KeyChangeDisplayName;
    var response=await http.post(url,body: {
      "phoneNumber":phoneNumber,
      "key":key,
      "displayName":displayName
    });


  }
}