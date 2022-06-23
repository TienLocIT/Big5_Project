import 'dart:convert';

import "package:http/http.dart" as http;
import '../utill/Verification.dart';
class changePhoneNumberNetwork{
  Future changPhoneNUmberMethod(String phoneNumber,String phoneNumberChange) async {
    final url="https://192.168.1.9:5000/users/changePhoneNumber";
    var response=await http.post(url,body: {
      "phoneNumber":phoneNumber,
      "phoneNumberChange":phoneNumberChange,
      "key":Validation.KeyChangePhoneNumber
    });
    if(response.statusCode==200){
      return jsonDecode(response.body);
    }
    else{
      throw Exception("Error");
    }
  }
}