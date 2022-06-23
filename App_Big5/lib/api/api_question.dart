// @dart=2.9
import 'dart:convert' as convert;
import 'package:http/http.dart' as http;
 question()async{
  String url="http://192.168.1.175:3000/questions";

  http.Response response=await http.get(url);
  var body=response.body;
  var data=convert.jsonDecode(body);
  var dataList=data.toList();
  return dataList;
}
questionPost(List answer,String id)async{
 String url="http://192.168.1.175:3000/questions?id=$id";
 var answerResult=[];
 for(var i=0;i<=answer.length-1;i++){
  String json=convert.jsonEncode(answer[i]);
  answerResult.add(json);
 }
 Map<String, String> headers = {"Content-type": "application/json"};
 final msg = convert.jsonEncode(answerResult);
 http.Response response=await http.post(url,headers:headers,body:msg);
 var body=response.body;
 var data=convert.jsonDecode(body);
 return data;
}