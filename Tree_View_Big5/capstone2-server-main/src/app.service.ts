import { Facet } from '@modules/suggest_questions/schemas/facet';
import { Keyword } from '@modules/suggest_questions/schemas/keywords';
import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';

@Injectable()
export class AppService {
    constructor(
        @InjectModel(Keyword.name)
        private readonly keyword: Model<Keyword>,
        @InjectModel(Facet.name)
        private readonly facet: Model<Facet>,
    ) {}
    getHello(): string {
        return 'Hello World!';
    }

    async getFacets() {
        return await this.facet.find({});
    }

    async getKeywords() {
        return await this.keyword.find({});
    }

    async convertData(keywords) {
        console.log('keywords', keywords);
        const x = await new this.keyword({ keywords }).save();
        console.log('X', x);
        return x;
    }
    async convertFacet(facet) {
        console.log('facet1', facet);
        const x = await new this.facet({ facet }).save();
        console.log('facet2', x);
        return x;
    }
}
