// @dart=2.9
import 'dart:async';
import 'package:animated_text_kit/animated_text_kit.dart';
import 'package:connectivity/connectivity.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:hexcolor/hexcolor.dart';
import '../Network/changePhoneNumberNetwork.dart';
import '../Network/userNetwork.dart';
import '../model/userModel.dart';
import 'userProfile.dart';
import '../utill/UserSecureStorage.dart';
class changePhoneNumber extends StatefulWidget {

  const changePhoneNumber({Key key}) : super(key: key);

  @override
  _changePhoneNumberState createState() => _changePhoneNumberState();
}

class _changePhoneNumberState extends State<changePhoneNumber> {
  bool changeSucces;
  String textAlert="";
  Future<userModel> user;
  final _formKey=GlobalKey<FormState>();
  final Connectivity _connectivity = Connectivity();
  final _phoneNumberChange=TextEditingController();
  final _phoneNumber=UserSecureStorage.getUserValue();
  StreamSubscription<ConnectivityResult> _connectivitySubscription;
  bool connection;
  Future<void> initConnectivity() async {
    ConnectivityResult result = ConnectivityResult.none;
    try {
      result = await _connectivity.checkConnectivity();
    } on PlatformException catch (e) {
      print(e.toString());
    }
    if (!mounted) {
      return Future.value(null);
    }
    return _updateConnectionStatus(result);
  }
  @override
  void initState() {
    // TODO: implement initState
    super.initState();
    initConnectivity();
    _phoneNumber.then((value){
      setState(() {
        user=userNetwork().getUser(value);
      });
    });
    _connectivitySubscription =
        _connectivity.onConnectivityChanged.listen(_updateConnectionStatus);
  }
  @override
  Widget build(BuildContext context) {
    return RefreshIndicator(
      onRefresh: (){},
      child: Scaffold(
        body: SafeArea(
          child: ListView(
            children: [
              FutureBuilder<userModel>(
                future: user,
                builder: (BuildContext context,AsyncSnapshot<userModel> snapshot){
                  if(snapshot.hasData){
                    return Column(
                      mainAxisAlignment: MainAxisAlignment.start,
                      children: [
                        FirstView(context,snapshot),
                        TextFieldPhoneNumber(context),
                        submitButton(context),
                        SizedBox(height: 10,),
                        TextAlert(context)
                      ],
                    );
                  }
                  else{
                    return Container(
                      width:MediaQuery.of(context).size.width,
                      height:MediaQuery.of(context).size.height,
                      child:Center(
                        child:CircularProgressIndicator()
                      )
                    );
                  }

                },

              ),
            ],
          ),
        ),
      ),
    );
  }
  Widget FirstView(BuildContext context,AsyncSnapshot<userModel> snapshot){
    return Container(
      margin: EdgeInsets.only(top: 70),
      child: Column(
        children: [
          Container(
            padding: EdgeInsets.fromLTRB(36.0, 0, 36.0, 25.0),
            child: Row(
              children: [
                Container(
                  margin: EdgeInsets.only(right: 10.0),
                  width: 50.0,
                  height: 50.0,
                  decoration: BoxDecoration(
                      boxShadow: [
                        BoxShadow(
                            color: Colors.grey.withOpacity(0.5),
                            spreadRadius: 5,
                            blurRadius: 7,
                            offset: Offset(0,3)
                        )
                      ],
                      image: DecorationImage(
                        image: AssetImage("assets/image/user.png"),
                        fit: BoxFit.cover,
                      ),
                      borderRadius: BorderRadius.circular(150.0),
                      border: Border.all(
                        color: Colors.white,
                        width: 5.0,
                      )
                  ),
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text("Welcome",style: GoogleFonts.roboto(
                        textStyle: TextStyle(
                            fontWeight: FontWeight.w700,
                            fontSize: 18,
                            color: HexColor("#9F9F9F")
                        )
                    ),
                    ),
                    Text(snapshot.data.displayName,style: GoogleFonts.roboto(
                        textStyle: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.w600,
                            color: HexColor("#3C3838")
                        )
                    ),)
                  ],
                ),
              ],
            ),
          ),
          Container(
            padding: EdgeInsets.fromLTRB(27, 0, 27, 50),
            child: Divider(
              thickness: 1,
              color: HexColor("#9F9F9F"),
            ),
          )
        ],
      ),
    );
  }
  Widget TextFieldPhoneNumber(BuildContext context){
    return Container(
      margin: EdgeInsets.only(top: 25),
      padding: EdgeInsets.only(left: 36,right: 36),
      child: Center(
        child: Form(
          key: _formKey,
          child: TextFormField(
            keyboardType: TextInputType.number,
            controller: _phoneNumberChange,
            maxLength: 11,
            validator: (value){
              if(value==null){
                return "Please enter your display name";
              }
              else{
                return null;
              }
            },
            style: GoogleFonts.roboto(
                textStyle: TextStyle(
                    fontSize: 18
                )
            ),

            decoration: InputDecoration(
              counterText: "",
                focusedBorder: UnderlineInputBorder(
                    borderSide: BorderSide(color: HexColor("#29BB89"),width: 1.0,style: BorderStyle.solid)
                ),
                enabledBorder: UnderlineInputBorder(
                    borderSide: BorderSide(color: Colors.black,width: 1.0,style: BorderStyle.solid)
                ),
                hintText: "Your phone number",
                hintStyle: GoogleFonts.roboto(textStyle: TextStyle(fontSize: 18,fontWeight: FontWeight.w500,color: HexColor("#9F9F9F").withOpacity(0.5)))
            ),
          ),
        ),
      ),
    );
  }
  Widget submitButton(BuildContext context){
    return Container(
      margin: EdgeInsets.only(top:50),
      width: MediaQuery.of(context).size.width,
      child: Center(
        child: Container(
          height: 50,
          width: 100,
          child: ElevatedButton(
            onPressed: (){
              if(_formKey.currentState.validate()){
                if(connection==true){
                  submitChangeDisplay();
                }
                else{
                  setState(() {
                    changeSucces=false;
                    textAlert="Please check connectivity";
                  });
                }
              }
            },
            style: ButtonStyle(
                foregroundColor: MaterialStateProperty.all<Color>(Colors.white),
                backgroundColor: MaterialStateProperty.all<Color>(HexColor("#2FBB89").withOpacity(0.5)),
                shape: MaterialStateProperty.all<RoundedRectangleBorder>(
                    RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12.0),
                    )
                )
            ),
            child: Text("Submit",style: GoogleFonts.roboto(
                textStyle: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 18
                )
            ),),
          ),
        ),
      ),
    );
  }
  Widget TextAlert(BuildContext context){
    if(changeSucces==true){
      return Container(
        child: Center(
          child: DefaultTextStyle(
              style: TextStyle(
                  color: Colors.red,
                  fontSize: 16,
                  fontWeight: FontWeight.w300
              ),
              child:AnimatedTextKit(
                  animatedTexts: [
                    WavyAnimatedText(textAlert,speed: Duration(milliseconds: 50)),
                  ],
                  isRepeatingAnimation: true,
                  repeatForever: true
              )
          ),
        ),
      );
    }
    else{
      return Container(
        child: Center(
          child: Text(
            textAlert,style: GoogleFonts.roboto(
              textStyle: TextStyle(
                  color: Colors.red,
                  fontSize: 16,
                  fontWeight: FontWeight.w300
              )
          ),
          ),
        ),
      );
    }

  }
  Future submitChangeDisplay(){
    _phoneNumber.then((value){
      Future changePhoneNumberDone=changePhoneNumberNetwork().changPhoneNUmberMethod(value, _phoneNumberChange.text);
      setState(() {
        changeSucces=true;
        textAlert="Please wait a few seconds....";
      });
      changePhoneNumberDone.then((value){
        if(value["error"] == "No"){
          UserSecureStorage.setUserValue(_phoneNumberChange.text);
          Navigator.push(context, MaterialPageRoute(builder: (context)=>userProfile()));
        }
        else{
          setState(() {
            changeSucces=false;
            textAlert=value["error"];
          });
        }
      });
    });

  }
  Future<void> _updateConnectionStatus(ConnectivityResult result) async{
    switch(result){
      case ConnectivityResult.wifi:
      case ConnectivityResult.mobile:
        setState(() {
          connection=true;
        });
        break;
      default:
        setState(() {
          connection=false;
        });
        break;
    }
  }
}
