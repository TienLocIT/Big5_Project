// @dart=2.9
import 'package:flutter_spinkit/flutter_spinkit.dart';
import "package:flutter/material.dart";
import '../api/api_question.dart';
class Loading extends StatefulWidget {
  const Loading({Key key}) : super(key: key);

  @override
  _LoadingState createState() => _LoadingState();
}

class _LoadingState extends State<Loading> {
  String time="loading";
  void setUpWorldTime() async{
    var data=await question();
    Map id={};
    id=ModalRoute.of(context).settings.arguments;
    Navigator.pushReplacementNamed(context, '/question',arguments: {
      "data":data,
      "id":id["_id"],
    });
  }
  @override
  void initState(){
    super.initState();
    setUpWorldTime();
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.blue,
      body: SafeArea(
          child:SpinKitCircle(
            color: Colors.white,
            size: 80.0,
          )
      ),
    );
  }
}
