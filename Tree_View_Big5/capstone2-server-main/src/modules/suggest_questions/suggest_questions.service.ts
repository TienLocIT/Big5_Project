import { HttpException, HttpStatus, Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { CreateSuggestQuestionDto } from './dto/create-suggest_question.dto';
import { UpdateSuggestQuestionDto } from './dto/update-suggest_question.dto';
import {
    Suggest_Question,
    SuggestQuestionDocument,
} from './schemas/suggestQuestion.schema';
import { Model } from 'mongoose';
import { Question } from './schemas/question';
import fetch from 'node-fetch';
import { User } from './schemas/user';
const axios = require('axios').default;

@Injectable()
export class SuggestQuestionsService {
    constructor(
        @InjectModel(Suggest_Question.name)
        private readonly model: Model<SuggestQuestionDocument>,
        @InjectModel(Question.name)
        private readonly question: Model<Question>,
        @InjectModel(User.name)
        private readonly user: Model<User>,
    ) {}

    async create(createSuggestQuestionDto: Question) {
        const { id } = createSuggestQuestionDto;
        const questionsFounded = await this.question.findOne({ id }).exec();
        if (questionsFounded) {
            throw new HttpException(
                {
                    status: HttpStatus.CONFLICT,
                    error: 'This suggested question is already exists in questions',
                },
                HttpStatus.CONFLICT,
            );
        }
        const x = await new this.question(createSuggestQuestionDto).save();
        console.log('XXXX', x);
        return await this.question.findByIdAndUpdate(
            { _id: x._id },
            { id: x._id },
        );
    }

    async generateSuggestQuestions(payload: any) {
        let res: any;
        console.log('payload', payload);

        try {
            await axios
                .post('http://127.0.0.1:8000/generate-questions', payload.body)
                .then((x) => {
                    res = x.data;
                });
            console.log('res', res.data);
            if (res) {
                const data = await this.model.insertMany(res.data);
                console.log('data', data);
                return data;
            }
        } catch (err) {
            console.log('err1', err);
        }
    }

    async saveAndSendToAllUser(createSuggestQuestionDto: Question) {
        const { id, question } = createSuggestQuestionDto;
        // const questionsFounded = await this.question.findOne({ id }).exec();

        console.log('createSuggestQuestionDto', createSuggestQuestionDto);

        const data = {
            key: 'MBe2IdLb-9dhasuidad02316156156552352sdggsdgsdaga2352bgdasasfas',
            question,
        };
        try {
            // if (!questionsFounded) {
            //     await new this.question({ ...createSuggestQuestionDto, }).save();
            // }
            await this.question.updateOne(
                { id },
                {
                    ...createSuggestQuestionDto,
                },
                { upsert: true },
            );
            await fetch(
                'http://greenbig5.herokuapp.com/expert/sendQuestionToAll',
                {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data),
                },
            );
        } catch (err) {
            throw new HttpException(
                {
                    status: HttpStatus.INTERNAL_SERVER_ERROR,
                    error: 'Internal Server Error!',
                },
                HttpStatus.INTERNAL_SERVER_ERROR,
            );
        }
    }

    async findSomeByKeywordAndFacet({ keyword, facet }) {
        const x = await this.model.find({
            big5EnvIndicator: `${keyword} ${facet}`,
        });
        return x;
    }

    getFacets() {
        return;
    }

    update(id: number, updateSuggestQuestionDto: UpdateSuggestQuestionDto) {
        return `This action updates a #${id} suggestQuestion`;
    }

    async remove(id: string) {
        return await this.model.findByIdAndDelete({
            _id: id,
        });
    }
}
