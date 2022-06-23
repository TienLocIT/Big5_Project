import {
    Controller,
    Get,
    Post,
    Body,
    Patch,
    Param,
    Delete,
    Query,
} from '@nestjs/common';
import { SuggestQuestionsService } from './suggest_questions.service';
import { CreateSuggestQuestionDto } from './dto/create-suggest_question.dto';
import { UpdateSuggestQuestionDto } from './dto/update-suggest_question.dto';
import { GetSuggestQuestionDto } from './dto/get-suggest_question.dto';
import { Question } from './schemas/question';

@Controller('suggest-questions')
export class SuggestQuestionsController {
    constructor(
        private readonly suggestQuestionsService: SuggestQuestionsService,
    ) {}

    @Post()
    async create(@Body() createSuggestQuestionDto: Question) {
        const x = await this.suggestQuestionsService.create(
            createSuggestQuestionDto,
        );
        return x;
    }

    @Post('generate-suggest-questions')
    async generateSuggestQuestions(@Body() payload: Question) {
        const x = await this.suggestQuestionsService.generateSuggestQuestions(
            payload,
        );
        return x;
    }

    @Post('save-and-send')
    async saveAndSendToAllUser(@Body() createSuggestQuestionDto: Question) {
        const x = await this.suggestQuestionsService.saveAndSendToAllUser(
            createSuggestQuestionDto,
        );
        return x;
    }

    @Get()
    findSomeByKeywordAndFacet(@Query() payload: GetSuggestQuestionDto) {
        return this.suggestQuestionsService.findSomeByKeywordAndFacet(payload);
    }

    @Get(':id')
    findOne(@Param('id') id: string) {
        // return this.suggestQuestionsService.findOne(+id);
    }

    @Patch(':id')
    update(
        @Param('id') id: string,
        @Body() updateSuggestQuestionDto: UpdateSuggestQuestionDto,
    ) {
        return this.suggestQuestionsService.update(
            +id,
            updateSuggestQuestionDto,
        );
    }

    @Delete(':id')
    async remove(@Param('id') id: string) {
        return await this.suggestQuestionsService.remove(id);
    }
}
