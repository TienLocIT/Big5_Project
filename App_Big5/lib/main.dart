
import 'package:flutter/material.dart';
import './pages/Question.dart';
import './pages/defindQuestion.dart';
import './pages/loginPages.dart';
import './utill/UserSecureStorage.dart';
import 'pages/defindQuestionStart.dart';
void main() {
  WidgetsFlutterBinding.ensureInitialized();
  final _phoneNumber=UserSecureStorage.getUserValue();
  _phoneNumber.then((value){

    if(value!=null){
      runApp(MaterialApp(
        home: Question(),
      ));
    }
    else {
      runApp(MaterialApp(
        home: defindQuestionStart(),
      ));
    }
  });
}




