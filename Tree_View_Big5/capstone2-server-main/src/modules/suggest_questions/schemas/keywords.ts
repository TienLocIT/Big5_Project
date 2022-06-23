import { Prop, raw, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document, Schema as sc } from 'mongoose';
export type KeywordDocument = Keyword & Document;

@Schema({ timestamps: true })
export class Children {
    @Prop({ type: String })
    id: string;

    @Prop({ type: Object })
    keyword: Object[];
}

@Schema({ timestamps: true })
export class KeywordType {
    @Prop({ type: String })
    id: string;

    @Prop({ type: String })
    title: string;

    @Prop({ type: Boolean })
    expanded: string;

    @Prop({ type: Object })
    children: Object[];
}

@Schema({ timestamps: true })
export class Keyword {
    @Prop()
    id: string;

    @Prop([KeywordType])
    keywords: KeywordType[];
}

export const KeywordSchema = SchemaFactory.createForClass(Keyword);
