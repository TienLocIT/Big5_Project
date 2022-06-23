import "package:http/http.dart" as http;
import '../utill/Verification.dart';
class answerNetwork{
  Future postAnswer(String phoneNumber,String id,String question,String answer,String index,String dem) async{
    final url="https://192.168.1.9:5000/users/postAnswer";
    final key=Validation.KeyPostUser;
    final body={
      "phoneNumber":phoneNumber,
      "key":key,
      "id":id,
      "question":question,
      "answer":answer,
      "index":index,
      "dem":dem,
    };
    await http.post(url,body: body);
  }
}