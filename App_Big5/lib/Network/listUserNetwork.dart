import 'dart:convert';
import "package:http/http.dart" as http;
import '../model/userModel.dart';
import '../utill/Verification.dart';
class listUserNetwork{
  final key=Validation.KeyGetListUser;
  Future<List<userModel>> getListUser() async{
      final url="http://192.168.1.9:5000/users/getListUser?key=$key";
      var response=await http.get(url);
      if(response.statusCode==200){
        print(response.body);
        List jsonResponse=jsonDecode(response.body);
        return jsonResponse.map((e) => new userModel.fromJson(e)).toList();
      }
      else{
        throw Exception("Error");
      }
  }
}