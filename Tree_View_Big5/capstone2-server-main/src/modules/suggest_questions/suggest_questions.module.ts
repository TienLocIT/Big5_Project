import { Module } from '@nestjs/common';
import { SuggestQuestionsService } from './suggest_questions.service';
import { SuggestQuestionsController } from './suggest_questions.controller';
import { MongooseModule } from '@nestjs/mongoose';
import {
    Suggest_Question,
    SuggestQuestionSchema,
} from './schemas/suggestQuestion.schema';
import { Question, QuestionSchema } from './schemas/question';
import { User, UserSchema } from '@modules/suggest_questions/schemas/user';

@Module({
    imports: [
        MongooseModule.forFeature([
            { name: Suggest_Question.name, schema: SuggestQuestionSchema },
            { name: Question.name, schema: QuestionSchema },
            { name: User.name, schema: UserSchema },
        ]),
    ],
    controllers: [SuggestQuestionsController],
    providers: [SuggestQuestionsService],
})
export class SuggestQuestionsModule {}
