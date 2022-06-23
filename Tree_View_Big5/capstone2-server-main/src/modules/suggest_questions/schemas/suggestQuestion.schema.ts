import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';
export type SuggestQuestionDocument = Suggest_Question & Document;

@Schema({ timestamps: true })
export class Suggest_Question {
    @Prop()
    id: string;

    @Prop()
    question: string;

    @Prop()
    big5EnvIndicator: string;

    @Prop()
    tags: string[];
}

export const SuggestQuestionSchema =
    SchemaFactory.createForClass(Suggest_Question);
