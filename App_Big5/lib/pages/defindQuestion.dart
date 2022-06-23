// @dart=2.9
import 'package:flutter/material.dart';
import '../model/userModel.dart';
import 'package:app_green_big5_capstone2/model/userModel.dart';
class DefineQuestionPage extends StatefulWidget {
  const DefineQuestionPage({Key key}) : super(key: key);

  @override
  _DefineQuestionPageState createState() => _DefineQuestionPageState();
}

class _DefineQuestionPageState extends State<DefineQuestionPage> {
  @override
  bool thankYou=false;
  Future<userModel> userInformation;
  Widget build(BuildContext context) {
    return WillPopScope(
        child: Scaffold(
          body: SafeArea(
            child: ListView(
              children: [
                Stack(
                 children: [

                 ],
                )
              ],
            ),
          ),
        ), onWillPop: ()async=>false);
  }
}
Widget FirstView(BuildContext context,AsyncSnapshot<userModel> snapshot){

}

