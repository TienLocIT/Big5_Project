// @dart=2.9
import 'dart:async';
import 'package:connectivity/connectivity.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:hexcolor/hexcolor.dart';
import '../Network/userNetwork.dart';
import '../model/userModel.dart';
import 'Question.dart';
import 'changeDisplayName.dart';
import 'changephoneNumber.dart';
import 'loginPages.dart';
import '../utill/UserSecureStorage.dart';
import 'failConnect.dart';
class userProfile extends StatefulWidget {
  const userProfile({ Key key}) : super(key: key);

  @override
  _userProfileState createState() => _userProfileState();
}

class _userProfileState extends State<userProfile> {
  bool connection;
  final Connectivity _connectivity = Connectivity();
  StreamSubscription<ConnectivityResult> _connectivitySubscription;
  Future<userModel> userInformation;

  final _phoneNumber = UserSecureStorage.getUserValue();

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
    _connectivitySubscription =
        _connectivity.onConnectivityChanged.listen(_updateConnectionStatus);
    _phoneNumber.then((value) {
      setState(() {
        userInformation = userNetwork().getUser(value);
      });
    });
  }

  @override
  void dispose() {
    // TODO: implement dispose
    _connectivitySubscription.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (connection == true) {
      return RefreshIndicator(
        onRefresh: _onRefresh,
        child: WillPopScope(
          onWillPop: () async => false,
          child: Scaffold(
            body: SafeArea(
              child: ListView(
                children: [
                  FutureBuilder<userModel>(
                    future: userInformation,
                    builder: (BuildContext context,
                        AsyncSnapshot<userModel> snapshot) {
                      if (snapshot.hasData) {
                        return Column(
                          children: [
                            FirstView(context, snapshot),
                            BottomView(context,snapshot)
                          ],
                        );
                      }
                      else {
                        return Container(
                          width: MediaQuery.of(context).size.width,
                          height: MediaQuery.of(context).size.height,
                          child: Center(
                              child:CircularProgressIndicator()
                          ),
                        );
                      }
                    },
                  ),

                ],
              ),
            ),
            bottomNavigationBar: BottomNavigationBar(
              items: [
                BottomNavigationBarItem(
                    icon: Container(height:20,child: Image.asset("assets/icons/Vector.png")),
                    label: ""),
                BottomNavigationBarItem(icon: Container(
                  height: 20,
                  child: Image.asset(
                      "assets/icons/Vector2.png", color: HexColor("#29BB89")),
                ),
                    label: "")
              ], onTap: (int index) => _navigateToScreens(index),
            ),
          ),
        ),
      );
    }
    else {
      return failConnect();
    }
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
  Widget BottomView(BuildContext context,AsyncSnapshot<userModel> snapshot){
    return Container(
      height: 150.0,
      padding: EdgeInsets.fromLTRB(30, 0, 20, 0),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Flexible(
                  flex: 1,
                  child: Container(height:20,child: Image.asset("assets/icons/phone.png",color: Colors.black,))),
              Flexible(
                flex: 2,
                child: Container(
                  alignment: Alignment.topLeft,
                  child: Text(snapshot.data.phoneNumber,
                    style: GoogleFonts.roboto(
                        textStyle: TextStyle(
                          fontWeight: FontWeight.w700,
                          fontSize: 18,
                          color: HexColor("#000000").withOpacity(0.7),
                        )
                    ),),
                ),
              ),
              Flexible(
                flex: 1,
                child: IconButton(
                  onPressed: (){
                  Navigator.push(context, MaterialPageRoute(builder:(context)=>changePhoneNumber()));
                  },
                  icon: Icon(Icons.arrow_forward_ios,color: Colors.black,),
                ),
              )
            ],
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Flexible(
                  flex: 1,
                  child: Image.asset("assets/icons/profilesetting.png",color: Colors.black,)),
              Flexible(
                flex: 2,
                child: Container(
                  alignment: Alignment.topLeft,
                  child: Text(snapshot.data.displayName,
                    style: GoogleFonts.roboto(
                        textStyle: TextStyle(
                          fontWeight: FontWeight.w700,
                          fontSize: 18,
                          color: HexColor("#000000").withOpacity(0.7),
                        )
                    ),),
                ),
              ),
              Flexible(
                flex: 1,
                child: IconButton(
                  onPressed: (){
                    Navigator.push(context, MaterialPageRoute(builder:(context)=>changeDisplayName()));
                  },
                  icon: Icon(Icons.arrow_forward_ios,color: Colors.black,),
                ),
              )
            ],
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Flexible(
                  flex: 1,
                  child: Image.asset("assets/icons/logout.png",color: Colors.black,)),
              Flexible(
                flex: 2,
                child: Container(
                  alignment: Alignment.topLeft,
                  child: Text("Logout",
                    style: GoogleFonts.roboto(
                        textStyle: TextStyle(
                          fontWeight: FontWeight.w700,
                          fontSize: 18,
                          color: HexColor("#000000").withOpacity(0.7),
                        )
                    ),),
                ),
              ),
              Flexible(
                flex: 1,
                child: IconButton(
                  onPressed: (){
                    logoutButton();
                  },
                  icon: Icon(Icons.arrow_forward_ios,color: Colors.black,),
                ),
              )
            ],
          ),

        ],
      ),
    );
  }
  void _navigateToScreens(int index){
     if(index==0){
      Navigator.push(context, MaterialPageRoute(
        builder: (context)=>Question(),
      ));
    }
  }
  void logoutButton(){
    showDialogLogout();
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
  Widget AlertDialogLogout(){

    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20.0)),
      insetPadding: EdgeInsets.all(10),
      child: Container(
        height: 178,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            SizedBox(height: 31.0,),
            Container(
              padding: EdgeInsets.only(left: 20,right: 20),
              child: Text("Do you want to logout?",style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w500,
                  color: HexColor("#000000")
              ),),
            ),
               SizedBox(height: 50,),
               Container(
                 padding: EdgeInsets.only(left: 30.0,right: 30.0),
                 child: Row(
                   mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    TextButton(
                        onPressed: (){
                          Navigator.pop(context);
                        },
                        child: Text("Cancel",style: TextStyle(
                          fontSize: 18,
                            fontWeight: FontWeight.w700,
                            color: HexColor("#29BB89")
                        ),
                        )
                    ),
                    ElevatedButton(
                        onPressed: (){
                          UserSecureStorage.deletePhoneNumber();
                          Navigator.push(context,MaterialPageRoute(
                              builder: (context)=>loginPage()
                          ));
                        },
                        style: ButtonStyle(
                            shape: MaterialStateProperty.all<RoundedRectangleBorder>(
                                RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(20.0),
                                )
                            ),
                          backgroundColor: MaterialStateProperty.all(HexColor("#29BB89"))
                        ),
                        child: Text("Submit",style: TextStyle(
                            fontWeight: FontWeight.w700,
                            fontSize: 18,
                            color: Colors.white,
                        ),
                        )
                    ),
                  ],
              ),
               ),
          ],
        ),
      ),
    );
  }
  void showDialogLogout(){
    showDialog(context: context,
        builder: (_)=>AlertDialogLogout(),
      barrierDismissible: false
    );
  }

  Future<void> _onRefresh() async{
    await Future.delayed(Duration(seconds: 5));
    _phoneNumber.then((value){
      setState(() {
       userInformation=userNetwork().getUser(value);
      });
    });
  }
}
