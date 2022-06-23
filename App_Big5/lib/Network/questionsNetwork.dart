// @dart=2.9
import 'package:http/http.dart' as http;
import '../model/userModel.dart';
import "../utill/Verification.dart";
class questionsNetwork{
  Future<userModel> postQuestionsAndGetUsers(String phoneNumber) async {
    var Key = Validation.KeyPost;
    var url = Uri.parse("https://192.168.1.9:5000/users/postQuestions");
    await http.post(url, body: {"phoneNumber": phoneNumber,"key": Key});
    // if(response.statusCode==200){
    //   return userModel.fromJson(jsonDecode(response.body));
    // }
    // else{
    //   throw Exception("Error");
    // }
  }
}