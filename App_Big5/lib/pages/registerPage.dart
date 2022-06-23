// @dart=2.9
import 'dart:async';
import 'package:animated_text_kit/animated_text_kit.dart';
import 'package:connectivity/connectivity.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:hexcolor/hexcolor.dart';
import '../Network/registerNetwork.dart';
import 'Question.dart';
import 'loginPages.dart';
import '../utill/UserSecureStorage.dart';
import 'failConnect.dart';
class registerPage extends StatefulWidget {
  const registerPage({Key key}) : super(key: key);

  @override
  _registerPageState createState() => _registerPageState();
}

class _registerPageState extends State<registerPage> {
  bool submitConnection;
  bool connection;
  final Connectivity _connectivity=Connectivity();
  StreamSubscription<ConnectivityResult> _connectivitySubscription;
  final _phoneNumber=TextEditingController();
  final _fullName=TextEditingController();
  final _displayName=TextEditingController();
  final _password=TextEditingController();
  final _repassword=TextEditingController();
  bool _autofocus=false;
  String textAlert="";
  String errorHave="";
  bool secureText=true;
  bool secureTextAnother=true;
  IconData iconPassword=Icons.visibility;
  IconData iconRePassword=Icons.visibility;
  final _formKey=GlobalKey<FormState>();
  Future<void> initConnectivity() async {
    ConnectivityResult result=ConnectivityResult.none;
    try{
      result=await _connectivity.checkConnectivity();
    } on PlatformException catch(e){
      print(e.toString());
    }
    if(!mounted){
      return Future.value(null);
    }
    return _updateConnectionStatus(result);
  }
  @override
  void initState() {
    // TODO: implement initState
    super.initState();
    initConnectivity();
    _connectivitySubscription=_connectivity.onConnectivityChanged.listen(_updateConnectionStatus);
  }
  @override
  void dispose() {
    // TODO: implement dispose
    _connectivitySubscription.cancel();
    super.dispose();
  }
  @override
  Widget build(BuildContext context) {
    if(connection==true){
      return WillPopScope(
        onWillPop: ()async=>false,
        child: GestureDetector(
          onTap: ()=>FocusScope.of(context).unfocus(),
          child: Scaffold(
            backgroundColor: Colors.white,
            body: ListView(
              children: [
                SafeArea(
                  child: SingleChildScrollView(
                    child: Container(
                      padding: EdgeInsets.fromLTRB(30, 20, 30, 20),
                      child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            firstView(context),
                            midView(context),
                            TextFieldView(context),
                            SizedBox(height: 10,),
                            TextAlert(context),
                            SizedBox(height: 30.0,),
                            BottomView(context),
                          ]
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    }
    else{
      return failConnect();
    }
  }
  Widget firstView(BuildContext context){
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          "Welcome,",
          style: TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 28.0
          ),
        ),
        Text(
          "Sign up to continue!",
          style: TextStyle(
            color: Colors.grey,
            fontSize: 25.0,
          ),
        ),
      ],
    );
  }
  Widget midView(BuildContext context){
    return Container(child: Center(child: Image(image: AssetImage("assets/image/logo_green.png"))));
  }
  Widget TextFieldView(BuildContext context){
    return Form(
      key: _formKey,
      child: Column(
        children: [
          TextFormField(
            controller: _fullName,
              validator: (value){
                if(value==null||value.isEmpty){
                  return 'Plese enter your name';
                }
                else{
                  return null;
                }
              },
              decoration: InputDecoration(
                counterText: "",
                labelStyle: TextStyle(
                ),
                border: OutlineInputBorder(
                  borderRadius:BorderRadius.all(Radius.circular(10.0)),
                ),
                hintText: "Fullname",
                labelText: "Fullname",
              )
          ),
          SizedBox(height: 10,),
          TextFormField(
              maxLength: 12,
              controller: _displayName,
              validator: (value){
                if(value==null||value.isEmpty){
                  return 'Plese enter your display name';
                }
                else{
                  return null;
                }
              },
              decoration: InputDecoration(
                counterText: "",
                labelStyle: TextStyle(
                ),
                border: OutlineInputBorder(
                  borderRadius:BorderRadius.all(Radius.circular(10.0)),
                ),
                hintText: "Displayname",
                labelText: "Displayname",
              )
          ),
          SizedBox(height: 10,),
          TextFormField(
            controller: _phoneNumber,
              validator: (value){
                if(value==null||value.isEmpty){
                  return 'Plese enter phone number';
                }
                else{
                  return null;
                }
              },
              maxLength: 11,
              keyboardType: TextInputType.number,
              decoration: InputDecoration(
                counterText: "",
                labelStyle: TextStyle(
                ),
                border: OutlineInputBorder(
                  borderRadius:BorderRadius.all(Radius.circular(10.0)),
                ),
                hintText: "Phone Number",
                labelText: "Phone Number",
              )
          ),
          SizedBox(height: 10,),
          TextFormField(
            maxLength: 15,
            controller: _password,
              validator: (value){
                if(value==null||value.isEmpty){
                  return 'Please enter password';
                }
                else{
                  return null;
                }
              },
              obscureText: secureText,
              decoration: InputDecoration(
                counterText: "",
                suffixIcon: IconButton(
                  icon: Icon(iconPassword),
                  onPressed: (){
                    securePassword(context);
                  },
                ),
                labelStyle: TextStyle(
                ),
                border: OutlineInputBorder(
                  borderRadius:BorderRadius.all(Radius.circular(10.0)),
                ),
                hintText: "Password",
                labelText: "Password",
              )
          ),
          SizedBox(height: 10,),
          TextFormField(
            autofocus: _autofocus,
            maxLength: 15,
            controller: _repassword,
              validator: (value){
                if(value==null||value.isEmpty){
                  return 'Please enter Re-password';
                }
                else{
                  return null;
                }
              },
              obscureText: secureTextAnother,
              decoration: InputDecoration(
                counterText: "",
                suffixIcon: IconButton(
                  icon: Icon(iconRePassword),
                  onPressed: (){
                      securePasswordAnother(context);
                  },
                ),
                labelStyle: TextStyle(
                ),
                border: OutlineInputBorder(
                  borderRadius:BorderRadius.all(Radius.circular(10.0)),
                ),
                hintText: "Re-Password",
                labelText: "Re-Password",
              )
          ),
        ],
      ),
    );
  }
  Widget BottomView(BuildContext context){
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Container(
          decoration: new BoxDecoration(
            boxShadow: [
              BoxShadow(
                color: Colors.white,
                blurRadius: 25.0, // soften the shadow
                spreadRadius: 7.0, //extend the shadow
                offset: Offset(
                  15.0, // Move to right horizontally
                  15.0, // Move to bottom Vertically
                ),
              )
            ],
          ),
          height: 60.0,
          child: ElevatedButton(
            onPressed: (){
              if(_formKey.currentState.validate()){
                if(connection==true){
                  SignUp(context);
                }
                else{
                  setState(() {
                    submitConnection=false;
                    textAlert="Please check connectivity...";
                  });
                }
                // ScaffoldMessenger.of(context).showSnackBar(
                //   const SnackBar(content: Text('Processing Data')),
                // );

              }
            },
            child: Text(
              "Sign Up",
              style: TextStyle(
                fontSize: 20.0,
              ),
            ),
            style: ButtonStyle(
                foregroundColor: MaterialStateProperty.all<Color>(Colors.white),
                backgroundColor: MaterialStateProperty.all<Color>(HexColor("#2FBB89")),
                shape: MaterialStateProperty.all<RoundedRectangleBorder>(
                    RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12.0),
                    )
                )
            ),
          ),
        ),
        SizedBox(height: 10.0,),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text("Have an account?",style:
            GoogleFonts.roboto(
                textStyle: TextStyle(
                  fontSize: 17,
                  fontWeight:FontWeight.w700,
                  color: HexColor("#C4C4C4"),
                )
            ),
            ),
            InkWell(
              onTap: (){
                Navigator.push(context, MaterialPageRoute(
                  builder: (context)=>loginPage()
                ));
              },
              child: Text(
                "Sign In",
                style: GoogleFonts.roboto(
                    textStyle: TextStyle(
                        color: HexColor("#29BB89"),
                        fontSize: 17,
                        fontWeight: FontWeight.w700
                    )
                ),
              ),
            )
          ],
        )
      ],
    );
  }
  Widget TextAlert(BuildContext context){
    if(submitConnection==true){
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
  void SignUp(BuildContext context){
    if(_password.text.trim()!=_repassword.text.trim()){
      setState(() {
        submitConnection=false;
        textAlert="Password does not match";
        _autofocus=true;
        _password.clear();
        _repassword.clear();
      });
    }
    else {
      Future error = postRegister(_phoneNumber.text, _fullName.text,_displayName.text, _password.text);
      setState(() {
        submitConnection=true;
        textAlert="Please wait a few seconds.....";
      });
      error.then((value) {
        setState(() {
          errorHave=value["error"];
        });
        if(errorHave=="No"){
          UserSecureStorage.setUserValue(_phoneNumber.text.trim());
          Navigator.push(context, MaterialPageRoute(
               builder: (context)=>Question()
           ));
        }else{
          setState(() {
            submitConnection=false;
            textAlert=value["error"];
          });
        }
      });
    }

    }
  void securePassword(BuildContext context){
    setState(() {
      if(secureText==true){
        iconPassword=Icons.visibility_off;
        secureText=false;
      }
      else{
        iconPassword=Icons.visibility;
        secureText=true;
      }
    });
  }
  void securePasswordAnother(BuildContext context){
    setState(() {
      if(secureTextAnother==true){
        iconRePassword=Icons.visibility_off;
        secureTextAnother=false;
      }
      else{
        iconRePassword=Icons.visibility;
        secureTextAnother=true;
      }
    });
  }
  Future postRegister(String phoneNumber,String fullName,String displayName,String password)=>
      registerNetwork().postRegister(phoneNumber, fullName,displayName, password);
  Future<void> _updateConnectionStatus(ConnectivityResult result) async{
    switch(result){
      case ConnectivityResult.wifi:
      case ConnectivityResult.mobile:
      case ConnectivityResult.none:
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
