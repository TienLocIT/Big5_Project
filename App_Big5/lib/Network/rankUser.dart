import 'package:http/http.dart' as http;
import 'dart:convert';

import '../utill/Verification.dart';

class rankUser{
  Future getRankUser(String phoneNumber) async{
      final key=Validation.KeyGetRankUser;
      String url="https://192.168.1.9:5000/users/getRankUser?phoneNumber=$phoneNumber&key=$key";
      var response=await http.get(url);
      if(response.statusCode==200){
        return jsonDecode(response.body);
      }
      else{
        throw Exception("Error");
      }
  }
}