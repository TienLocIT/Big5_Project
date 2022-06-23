import { Question } from '@modules/suggest_questions/schemas/question';
import { HttpException, HttpStatus, Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { CreateQuestionDto } from './dto/create-question.dto';
import { UpdateQuestionDto } from './dto/update-question.dto';
import { Model } from 'mongoose';
import fetch from 'node-fetch';
const axios = require('axios').default;

@Injectable()
export class QuestionsService {
    constructor(
        @InjectModel(Question.name)
        private readonly question: Model<Question>,
    ) {}
    async apiGetOutputModelResult(data: any) {
        let y: any = {};
        await axios.post('http://127.0.0.1:8000/predict', data).then((x) => {
            y = x.data;
        });
        console.log('XXX', y.OUTPUT_MODEL);
        return {
            outputModel: y.OUTPUT_MODEL,
        };
    }

    findAll() {
        return `This action returns all questions`;
    }

    async findSomeByKeywordAndFacet({ keyword, facet }) {
        if (!keyword || !facet) {
            const allQuestions: any = await this.question.find({});

            const allQuestionsAfterConvert = allQuestions.map((question) => {
                return {
                    question: question._doc.question,
                    id: question._doc.id,
                    point: question._doc.point,
                    personality: question._doc.personality,
                    numberUser: question._doc.numberUser || 0,
                    totalAnswer: question._doc.totalAnswer || 0,
                    outputModel: question._doc.outputModel || '?',
                    trait: question._doc.trait,
                    big5EnvIndicator: question._doc.big5EnvIndicator,
                    facet: question._doc.facet,
                    keyword: question._doc.keyword,
                };
            });
            console.log('allQuestionsAfterConvert', allQuestionsAfterConvert);
            return allQuestionsAfterConvert;
        } else {
            return await this.question.find({
                big5EnvIndicator: `${keyword} ${facet}`,
            });
        }
    }

    findOne(id: number) {
        return `This action returns a #${id} question`;
    }

    async updateQuestion(updateQuestionDto: UpdateQuestionDto) {
        // const id
        const { id } = updateQuestionDto;
        delete (updateQuestionDto as any).facet;
        delete (updateQuestionDto as any).keyword;
        const questionFounded = await this.question.findOne({ id }).exec();
        if (!questionFounded) {
            throw new HttpException(
                {
                    status: HttpStatus.NOT_FOUND,
                    error: 'This question is not exist',
                },
                HttpStatus.NOT_FOUND,
            );
        }
        await this.question.findOneAndUpdate({ id }, updateQuestionDto);
    }

    remove(id: number) {
        return `This action removes a #${id} question`;
    }
}
