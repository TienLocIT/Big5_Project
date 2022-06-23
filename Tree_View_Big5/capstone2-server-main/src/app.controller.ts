import { Controller, Get, Post } from '@nestjs/common';
import { AppService } from './app.service';
const axios = require('axios').default;
import { keywords, facet } from './test-data';
import { ObjectId } from 'bson';

@Controller()
export class AppController {
    constructor(private readonly appService: AppService) {}

    @Get()
    getHello(): string {
        return this.appService.getHello();
    }

    @Get('facets')
    async getFacets() {
        return await this.appService.getFacets();
    }
    @Get('keywords')
    async getKeywords() {
        return await this.appService.getKeywords();
    }

    convertToCapitalize(title: string) {
        const words = title.split(' ');
        return words
            .map(([firstChar, ...restChars]) => {
                return firstChar.toUpperCase() + restChars.join('');
            })
            .join(' ');
    }

    convertRecursive(keywords) {
        for (let i = 0; i < keywords.length; i++) {
            keywords[i].title = this.convertToCapitalize(keywords[i].title);
            keywords[i].id = keywords[i].id?.$oid
                ? keywords[i].id?.$oid
                : keywords[i].id
                ? keywords[i].id
                : new ObjectId();
            if (keywords[i].children && keywords[i].children.length) {
                this.convertRecursive(keywords[i].children);
            }
        }
    }

    @Post()
    async convertData(): Promise<any> {
        this.convertRecursive(keywords);
        return await this.appService.convertData(keywords);
    }

    @Post('facet')
    async convertFacet(): Promise<any> {
        this.convertRecursive(facet);
        return await this.appService.convertFacet(facet);
    }
}
