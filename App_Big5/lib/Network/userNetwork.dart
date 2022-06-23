// @dart=2.9
import 'dart:convert';
import "package:http/http.dart" as http;
import 'package:app_green_big5_capstone2/model/userModel.dart';
import "package:app_green_big5_capstone2/utill/Verification.dart";
class userNetwork{
  Future<userModel> getUser(String phoneNumber) async {
      final key=Validation.KeyGetUser;
      String url="http://192.168.1.9:5000/users/getUser?phoneNumber=$phoneNumber&key=$key";
      var response=await http.get(url);
      if(response.statusCode==200){
        return userModel.fromJson(jsonDecode(response.body));
      }
      else{
        throw Exception("error");
      }
  }
}