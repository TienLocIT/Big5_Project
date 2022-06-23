// @dart=2.9
class LoginRequestModel{
  String email,password;
  LoginRequestModel({this.email,this.password});
  Map<String,dynamic> toJson(){
    Map<String,dynamic> map={
      "email":email,
      "password":password
    };
    return map;
  }
}