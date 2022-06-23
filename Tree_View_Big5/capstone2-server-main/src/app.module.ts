import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { MongooseModule } from '@nestjs/mongoose';
import { SuggestQuestionsModule } from './modules/suggest_questions/suggest_questions.module';

import { QuestionsModule } from './modules/questions/questions.module';
import {
    Keyword,
    KeywordSchema,
} from '@modules/suggest_questions/schemas/keywords';
import { Facet, FacetSchema } from '@modules/suggest_questions/schemas/facet';

@Module({
    imports: [
        MongooseModule.forRoot(
            'mongodb+srv://GreenBig5:1QgvDJucHUpHxEbn@greenbig5.uszbe.mongodb.net/green_big_5?retryWrites=true&w=majority',
        ),
        MongooseModule.forFeature([
            { name: Keyword.name, schema: KeywordSchema },
            { name: Facet.name, schema: FacetSchema },
        ]),
        SuggestQuestionsModule,
        QuestionsModule,
    ],
    controllers: [AppController],
    providers: [AppService],
})
export class AppModule {}
