import { Prop, raw, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document, Schema as sc } from 'mongoose';
export type KeywordDocument = Facet & Document;

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
export class Facet {
    @Prop()
    id: string;

    @Prop([KeywordType])
    facet: KeywordType[];
}

export const FacetSchema = SchemaFactory.createForClass(Facet);
