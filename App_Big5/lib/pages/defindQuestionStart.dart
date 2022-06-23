// @dart=2.9
import 'package:flutter/material.dart';
import 'package:hexcolor/hexcolor.dart';
class defindQuestionStart extends StatelessWidget {
  const defindQuestionStart({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: ()async=>false,
      child: SafeArea(
        child: Scaffold(
          body: ListView(
              children: [
                Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                      Container(
                        child: Image(image: AssetImage("assets/image/logo_green.png"),),
                      ),
                     textWidget("Surprise!!!!"),
                    textWidget("You have a new question!!!"),
                    textWidget("Do you want answer the question?"),
                    yesnoView()
                  ],
                )
              ],
          ),
        ),
      ),
    );
  }
  Widget textWidget(String text){
    return Container(
      child: Text(text,style: TextStyle(
        fontSize: 14,
        fontWeight:FontWeight.w700,
        color: HexColor("#C4C4C4"),
      ),),
    );
  }
  Widget yesnoView(){
    return Row(
      children: [
        SizedBox(
          width: 20,
          height: 100,
          child: TextButton(
            child: Text("Yess"),
          ),
        )
      ],
    );
  }
}
