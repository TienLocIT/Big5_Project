// @dart=2.9
import 'dart:convert' as convert;
import 'package:http/http.dart' as http;
  Future login(String username,String password) async {
    String url="http://192.168.1.175:3000/user";
    Map<String, String> headers = {"Content-type": "application/json"};
    final msg = convert.jsonEncode({"username":username,"password":password});
    http.Response response=await http.post(url,headers:headers,body: msg);
    var body=response.body;
    var data=convert.jsonDecode(body);
    return data;
}
