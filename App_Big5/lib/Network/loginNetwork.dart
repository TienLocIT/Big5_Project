import 'dart:convert';

import 'package:http/http.dart' as http;
import '../utill/Verification.dart';
class loginNetwork{
  Future loginPostNetwork(String phoneNumber,String password) async{
    var Key = Validation.Key;
    var url = Uri.parse("https://192.168.1.9:5000/users/login");
    var response = await http.post(url, body: {"phoneNumber": phoneNumber, "password":password,"key": Key});
    if(response.statusCode==200){
        return jsonDecode(response.body);
    }
    else{
      throw Exception("Error");
    }
  }
}