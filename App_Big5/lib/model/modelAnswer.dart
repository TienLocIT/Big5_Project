// @dart=2.9
class ModelAnswer {
  String questions, answer;

  ModelAnswer({this.questions, this.answer});

  ModelAnswer.fromJson(Map<String, dynamic> json)
      : questions = json['questions'],
        answer = json['answer'];

  Map<String, dynamic> toJson() {
    return {
      'questions': questions,
      'answer': answer,
    };
  }
}