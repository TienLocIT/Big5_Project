// @dart=2.9
import 'package:flutter/material.dart';
import "package:lottie/lottie.dart";
class GetStarted extends StatefulWidget {
  const GetStarted({Key key}) : super(key: key);

  @override
  _GetStartedState createState() => _GetStartedState();
}

class _GetStartedState extends State<GetStarted> {
  @override
  Widget build(BuildContext context) {
    Map data={};
    data=ModalRoute.of(context).settings.arguments;
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Lottie.asset("assets/51971-hello.json"),
              Divider(
                height: 60.0,
                color: Colors.green,
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    "Green",
                    style: TextStyle(
                      color: Colors.green,
                      fontSize: 23.0
                    ),
                  ),
                  Text(
                    "Big5",
                    style: TextStyle(
                        color: Colors.black,
                        fontSize: 23.0
                    ),
                  )
                ],
              ),
              SizedBox(height: 20.0,),
              Text("We would like to give you the best experience"),
              Text("We can give"),
              SizedBox(height:40.0),
              Container(
                height: 50.0,
                child: ElevatedButton(
                  onPressed: (){
                    Navigator.pushReplacementNamed(context, "/loading",arguments: {
                      "_id":data
                    });
                  },
                  child: Text("Get Started"),
                  style: ButtonStyle(
                      foregroundColor: MaterialStateProperty.all<Color>(Colors.white),
                      backgroundColor: MaterialStateProperty.all<Color>(Colors.green),
                    shape: MaterialStateProperty.all<RoundedRectangleBorder>(
                        RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12.0),
                        )
                  ),
                ),
              )
              )
            ],
          ),
        ),
      ),
    );
  }
}
